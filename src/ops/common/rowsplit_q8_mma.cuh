#pragma once

// Q4/Q5 RowSplit x int8 activation GEMM for prefill (mma.m16n8k32.s8).
//
// Signed Q4 [-8, 7] and Q5 [-16, 15] codes are exact int8, and activations come from
// q8_quantize_act<K, Q8ActLayout::Q4Q5Lane> (per-256 absmax scales). K is permuted inside every
// 64-value group: quad lane q owns code bytes 8q..8q+7 of the group (values 16q..16q+15); the low
// nibbles of bytes 8q..8q+3 are its A operand for k-positions [4q, 4q+4) of the group's first
// k32 step and the high nibbles for [16+4q, 16+4q+4), bytes 8q+4..8q+7 likewise for the second
// step. Q4Q5Lane activations follow the same order, so a lane's B operands for a group are one
// 16-byte load and its A operands are a mask, a shift and a sign extension (plus the spread high
// bits for Q5). Both k32 steps of a group share the weight scale, so they accumulate inside the
// MMA; the int32 result is converted without I2F (the accumulator starts at the bits of 1.5*2^23
// and |sum| <= 16*127*64 < 2^22) and scaled by the group's FP16 weight scale, then by the token's
// activation scale once per 256 values.
//
// CTA = WM warps x 16*MT rows over 8*NT tokens; each warp loads its rows' codes straight into
// registers one 256-value stage ahead, the token tile's activations are shared through a 2-stage
// cp.async ring. grid = (ceil(T / (8*NT)), sum_j n_j / BM); every job's n must divide by BM.

#include "ops/common/device_class.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/q8_act.cuh"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

enum class RowSplitQ8Output : std::uint8_t {
    Store,       // job.out[t * out_ld + out_row_offset + row] = value
    ResidualAdd, // job.out[t * out_ld + out_row_offset + row] += value
};

constexpr int kRowSplitQ8MagicInt   = 0x4B400000;
constexpr float kRowSplitQ8MagicF32 = 12582912.0f;

// Spreads bits 0, 2, 4, 6 of x to bit 0 of bytes 0..3.
__device__ __forceinline__ unsigned rowsplit_q8_spread_even(unsigned x) {
    return ((x & 0x55u) * 0x41041u) & 0x01010101u;
}

__device__ __forceinline__ unsigned rowsplit_q8_sext4(unsigned v) {
    return v | ((v & 0x08080808u) * 0x1fu);
}

__device__ __forceinline__ unsigned rowsplit_q8_sext5(unsigned v) {
    return v | ((v & 0x10101010u) * 0x0fu);
}

template <int K, int MT, int NT, int WM, int MB, int Jobs, RowSplitGroupedMmaCodec Codec,
          RowSplitQ8Output Output>
__global__ __launch_bounds__(WM * 32, MB) void rowsplit_q8_mma_kernel(
    const std::int8_t* __restrict__ xq, const float* __restrict__ sx, int sx_stride,
    RowSplitGroupedMmaJob job0, RowSplitGroupedMmaJob job1, RowSplitGroupedMmaJob job2,
    RowSplitGroupedMmaJob job3, int tokens) {
    constexpr int kGroups   = K / 64;          // weight groups per row
    constexpr int kStages   = K / kQ8ActGroup; // 256-value stages
    constexpr int kThreads  = WM * 32;
    constexpr int BM        = 16 * MT * WM;
    constexpr int BN        = 8 * NT;
    constexpr int kXBytes   = BN * kQ8ActGroup;
    constexpr int kStageB   = kXBytes + BN * 4;
    static_assert(K % kQ8ActGroup == 0 && kQ8ActTokenPad % BN == 0, "unsupported tile");
    static_assert(Jobs >= 1 && Jobs <= 4, "one to four jobs");

    __shared__ __align__(16) std::uint8_t smem[2][kStageB];

    // Row tile -> job.
    int tile = static_cast<int>(blockIdx.y);
    RowSplitGroupedMmaJob job = job0;
    if constexpr (Jobs > 1) {
        const RowSplitGroupedMmaJob jobs[3] = {job1, job2, job3};
        int tiles = job0.n / BM;
#pragma unroll
        for (int j = 0; j < Jobs - 1; ++j) {
            if (tile >= tiles) {
                tile -= tiles;
                job   = jobs[j];
                tiles = jobs[j].n / BM;
            }
        }
    }
    const bool q5 = Codec == RowSplitGroupedMmaCodec::Q5 ||
                    (Codec == RowSplitGroupedMmaCodec::Mixed && job.q5);

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int token0 = static_cast<int>(blockIdx.x) * BN;
    const int row0   = tile * BM + warp * 16 * MT; // job-relative

    const auto stage = [&](int slot, int s) {
        std::uint8_t* xs = smem[slot];
        for (int i = tid; i < BN * 16; i += kThreads) {
            const int t = i >> 4, chunk = i & 15;
            const bool valid = token0 + t < tokens;
            const std::int8_t* src = xq + static_cast<std::int64_t>(valid ? token0 + t : 0) * K +
                                     s * kQ8ActGroup + chunk * 16;
            cp_async_zfill<16, Cache::cg>(xs + t * kQ8ActGroup + ((chunk ^ ((t & 1) << 2)) * 16),
                                          src, valid ? 16 : 0);
        }
        for (int i = tid; i < BN / 4; i += kThreads) {
            cp_async_zfill<16, Cache::cg>(xs + kXBytes + i * 16,
                                          sx + static_cast<std::int64_t>(s) * sx_stride + token0 +
                                              i * 4,
                                          16);
        }
        cp_commit();
    };

    struct Raw {
        uint2 codes[4];
        unsigned high[4];
        uint2 scales;
    };
    const std::uint8_t* row_codes[MT][2];
    const std::uint8_t* row_high[MT][2];
    const std::uint8_t* row_scales[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const std::int64_t row = row0 + mi * 16 + gid + r * 8;
            row_codes[mi][r]       = job.codes + row * kGroups * 32 + lid * 8;
            row_high[mi][r]        = q5 ? job.high + row * kGroups * 8 + lid * 2 : nullptr;
            row_scales[mi][r]      = job.scales + row * kGroups * 2;
        }
    }
    const auto load = [&](int s, Raw (&raw)[MT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    raw[mi][r].codes[g] = __ldg(
                        reinterpret_cast<const uint2*>(row_codes[mi][r] + (4 * s + g) * 32));
                    raw[mi][r].high[g] =
                        q5 ? __ldg(reinterpret_cast<const unsigned short*>(row_high[mi][r] +
                                                                           (4 * s + g) * 8))
                           : 0u;
                }
                raw[mi][r].scales =
                    __ldg(reinterpret_cast<const uint2*>(row_scales[mi][r] + s * 8));
            }
        }
    };

    stage(0, 0);
    Raw cur[MT][2], next[MT][2];
    load(0, cur);

    float acc[MT][NT][4] = {};
#pragma unroll 1
    for (int s = 0; s < kStages; ++s) {
        const bool more = s + 1 < kStages;
        if (more) {
            load(s + 1, next);
            stage((s + 1) & 1, s + 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const std::uint8_t* xs = smem[s & 1];
        const std::uint8_t* xrow = xs + gid * kQ8ActGroup;
        const float* srow = reinterpret_cast<const float*>(xs + kXBytes) + lid * 2;
        float gacc[MT][NT][4];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            unsigned a0[MT][4], a1[MT][4]; // first and second k32 step
            float ws[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const uint2 c = cur[mi][r].codes[g];
                    unsigned lo0 = c.x & 0x0f0f0f0fu, hi0 = (c.x >> 4) & 0x0f0f0f0fu;
                    unsigned lo1 = c.y & 0x0f0f0f0fu, hi1 = (c.y >> 4) & 0x0f0f0f0fu;
                    if (q5) {
                        const unsigned h = cur[mi][r].high[g];
                        lo0 = rowsplit_q8_sext5(lo0 | (rowsplit_q8_spread_even(h) << 4));
                        hi0 = rowsplit_q8_sext5(hi0 | (rowsplit_q8_spread_even(h >> 1) << 4));
                        lo1 = rowsplit_q8_sext5(lo1 | (rowsplit_q8_spread_even(h >> 8) << 4));
                        hi1 = rowsplit_q8_sext5(hi1 | (rowsplit_q8_spread_even(h >> 9) << 4));
                    } else {
                        lo0 = rowsplit_q8_sext4(lo0);
                        hi0 = rowsplit_q8_sext4(hi0);
                        lo1 = rowsplit_q8_sext4(lo1);
                        hi1 = rowsplit_q8_sext4(hi1);
                    }
                    a0[mi][r]     = lo0;
                    a0[mi][r + 2] = hi0;
                    a1[mi][r]     = lo1;
                    a1[mi][r + 2] = hi1;
                    const unsigned sbits = (g < 2 ? cur[mi][r].scales.x : cur[mi][r].scales.y) >>
                                           (16 * (g & 1));
                    ws[mi][r] = __half2float(__ushort_as_half(static_cast<unsigned short>(sbits)));
                }
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int t = ni * 8 + gid;
                const uint4 xb = *reinterpret_cast<const uint4*>(
                    xrow + ni * 8 * kQ8ActGroup + (((g * 4 + lid) ^ ((t & 1) << 2)) * 16));
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    int p0 = kRowSplitQ8MagicInt, p1 = kRowSplitQ8MagicInt;
                    int p2 = kRowSplitQ8MagicInt, p3 = kRowSplitQ8MagicInt;
                    mma_s8(p0, p1, p2, p3, a0[mi][0], a0[mi][1], a0[mi][2], a0[mi][3], xb.x, xb.y);
                    mma_s8(p0, p1, p2, p3, a1[mi][0], a1[mi][1], a1[mi][2], a1[mi][3], xb.z, xb.w);
                    const float v0 = (__int_as_float(p0) - kRowSplitQ8MagicF32) * ws[mi][0];
                    const float v1 = (__int_as_float(p1) - kRowSplitQ8MagicF32) * ws[mi][0];
                    const float v2 = (__int_as_float(p2) - kRowSplitQ8MagicF32) * ws[mi][1];
                    const float v3 = (__int_as_float(p3) - kRowSplitQ8MagicF32) * ws[mi][1];
                    if (g == 0) {
                        gacc[mi][ni][0] = v0;
                        gacc[mi][ni][1] = v1;
                        gacc[mi][ni][2] = v2;
                        gacc[mi][ni][3] = v3;
                    } else {
                        gacc[mi][ni][0] += v0;
                        gacc[mi][ni][1] += v1;
                        gacc[mi][ni][2] += v2;
                        gacc[mi][ni][3] += v3;
                    }
                }
            }
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const float2 xs2 = *reinterpret_cast<const float2*>(srow + ni * 8);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                acc[mi][ni][0] = fmaf(gacc[mi][ni][0], xs2.x, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf(gacc[mi][ni][1], xs2.y, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf(gacc[mi][ni][2], xs2.x, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf(gacc[mi][ni][3], xs2.y, acc[mi][ni][3]);
            }
        }
        if (more) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                cur[mi][0] = next[mi][0];
                cur[mi][1] = next[mi][1];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int r0 = row0 + mi * 16 + gid;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int t = token0 + ni * 8 + lid * 2;
            const auto put = [&](int token, int row, float value) {
                if (token >= tokens) return;
                __nv_bfloat16* p = job.out + static_cast<std::int64_t>(token) * job.out_ld +
                                   job.out_row_offset + row;
                if constexpr (Output == RowSplitQ8Output::ResidualAdd) {
                    value += __bfloat162float(*p);
                }
                *p = __float2bfloat16_rn(value);
            };
            put(t, r0, acc[mi][ni][0]);
            put(t + 1, r0, acc[mi][ni][1]);
            put(t, r0 + 8, acc[mi][ni][2]);
            put(t + 1, r0 + 8, acc[mi][ni][3]);
        }
    }
}

// Job for rows [row_begin, row_begin + rows) of a Q4/Q5 RowSplit weight, written to rows
// [out_row_offset, out_row_offset + rows) of a token-major BF16 output with leading dimension out_ld.
inline RowSplitGroupedMmaJob rowsplit_q8_job(const Weight& weight, std::int32_t row_begin,
                                             std::int32_t rows, __nv_bfloat16* out,
                                             std::int32_t out_ld, std::int32_t out_row_offset) {
    if (row_begin < 0 || rows <= 0 || row_begin + rows > weight.n || weight.group != 64) {
        throw std::invalid_argument("rowsplit int8 GEMM: invalid weight row view");
    }
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const bool q5             = weight.qtype == QType::Q5_G64_FP16;
    return RowSplitGroupedMmaJob{
        static_cast<const std::uint8_t*>(weight.qdata) + row_begin * groups * 32,
        q5 ? static_cast<const std::uint8_t*>(weight.qhigh) + row_begin * groups * 8 : nullptr,
        static_cast<const std::uint8_t*>(weight.scales) + row_begin * groups * 2,
        out,
        rows,
        out_ld,
        out_row_offset,
        q5,
    };
}

// MT=1, NT=8, 4 warps (64 rows x 64 tokens), three CTAs per SM. Small-SM cards (36-SM RTX 5060 Ti)
// run two: whole-model prefill pp163 1091 -> 1106 tok/s, pp1024 1546 -> 1581 tok/s.
template <int K, int Jobs, RowSplitGroupedMmaCodec Codec, RowSplitQ8Output Output>
void rowsplit_q8_launch(const Q8Act& act, const RowSplitGroupedMmaJob (&jobs)[4], int tokens,
                        cudaStream_t stream) {
    constexpr int MT = 1, NT = 8, WM = 4;
    constexpr int BM = 16 * MT * WM, BN = 8 * NT;
    int tiles = 0;
    for (int j = 0; j < Jobs; ++j) {
        if (jobs[j].n <= 0 || jobs[j].n % BM != 0) {
            throw std::invalid_argument("rowsplit int8 GEMM: job rows must be a positive multiple of 64");
        }
        tiles += jobs[j].n / BM;
    }
    const dim3 grid(static_cast<unsigned>((tokens + BN - 1) / BN), static_cast<unsigned>(tiles));
    if (small_sm_device()) {
        rowsplit_q8_mma_kernel<K, MT, NT, WM, 2, Jobs, Codec, Output><<<grid, WM * 32, 0, stream>>>(
            act.xq, act.sx, act.sx_stride, jobs[0], jobs[1], jobs[2], jobs[3], tokens);
    } else {
        rowsplit_q8_mma_kernel<K, MT, NT, WM, 3, Jobs, Codec, Output><<<grid, WM * 32, 0, stream>>>(
            act.xq, act.sx, act.sx_stride, jobs[0], jobs[1], jobs[2], jobs[3], tokens);
    }
}

} // namespace ninfer::ops::detail
