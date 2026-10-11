#pragma once

// IQ3_XXS x BF16 matrix kernels for the dense MLP (gate/up LinearSwiGLU and down LinearAdd).
//
// Both kernels use mma.m16n8k16 with the weights as the A operand (16-row tiles) and contract one
// 256-value block per warp step; signed grid magnitudes are exact BF16 integers and each 32-value
// sub-block scale is applied to its FP32 partial. K is permuted inside every sub-block (the dot
// product is order independent): MMA k-index 2*q + {0,1} of half h is value 8*q + 4*h + {0,1} and
// k-index 2*q + 8 + {0,1} is value 8*q + 4*h + {2,3}, where q is the lane's position in its quad.
// A lane's eight values per row are then the two whole grid entries 2q and 2q+1, which the
// iq3_xxs_planar_v1 block order stores contiguously per quad lane (see tools/artifact/codecs/
// iq3_xxs.py), and its eight activations per token are one 16-byte load.
//
// Direct kernel (small T): every lane loads its 24 code bytes of a row-block straight into
// registers, double-buffered across the warp's strided K blocks; the only CTA barrier is the final
// cross-warp K reduction. Staged kernel (prefill): cp.async stages 16*MT rows per K tile and the
// row tiles are reused across token tiles of 8*NT.
//
// SwiGLU tiles are 8 gate rows followed by the matching 8 up rows, so one thread holds both
// projections of an output.

#include "core/arena.h"
#include "core/device.h" // CUDA_CHECK
#include "ops/common/iq3xxs.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/q8_act.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Iq3XxsSwiGluOut {
    __nv_bfloat16* out;     // [intermediate, tokens] token-major
    std::int32_t half_rows; // intermediate rows; up row = gate row + half_rows

    // Every 16-row MMA tile is 8 gate rows followed by the matching 8 up rows.
    __device__ __forceinline__ int weight_row(int output_row0, int local) const {
        return output_row0 + (local >> 4) * 8 + (local & 7) + ((local & 8) ? half_rows : 0);
    }
};

struct Iq3XxsResidualOut {
    __nv_bfloat16* residual; // [rows, tokens], leading dimension `stride`
    std::int64_t stride;

    __device__ __forceinline__ void add(int row, int token, float value) const {
        __nv_bfloat16* p = residual + static_cast<std::int64_t>(token) * stride + row;
        *p               = __float2bfloat16_rn(value + __bfloat162float(*p));
    }
};

// One quad lane's share of one row-block.
struct Iq3XxsLaneCodes {
    uint4 index;          // index bytes of entries (2q, 2q+1) for sub-blocks 0..7
    std::uint64_t signs;  // bits [7s, 7s+7): sign index q of sub-block s
    std::uint32_t scales; // 4-bit scale of sub-block s at bits [4s, 4s+4)
    float d;
};

struct Iq3XxsTables {
    uint2 grid[256];     // BF16x2 magnitudes (0,1) and (2,3) of every grid entry
    uint2 sign_mask[16]; // BF16 sign masks for a 4-bit nibble: bits (0,1) -> .x, (2,3) -> .y

    __device__ __forceinline__ void load() {
        iq3xxs_load_grid_bf16(grid);
        const int t = static_cast<int>(threadIdx.x);
        if (t < 16) {
            sign_mask[t] = make_uint2(iq3xxs_bf16x2_sign_mask(static_cast<unsigned>(t) & 3u),
                                      iq3xxs_bf16x2_sign_mask(static_cast<unsigned>(t) >> 2));
        }
    }
};

// Signed BF16 fragments of the lane's two grid entries of sub-block S: frag[h] = {k-index 2q pair,
// k-index 2q + 8 pair} of half h.
template <int S>
__device__ __forceinline__ void iq3xxs_lane_fragments(const Iq3XxsTables& t,
                                                      const Iq3XxsLaneCodes& c, uint2 (&frag)[2]) {
    const unsigned word  = S < 2 ? c.index.x : S < 4 ? c.index.y : S < 6 ? c.index.z : c.index.w;
    const unsigned pair  = word >> (16 * (S & 1));
    const unsigned signs = iq3xxs_sign8(static_cast<unsigned>(c.signs >> (7 * S)) & 127u);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint2 entry = t.grid[(pair >> (8 * h)) & 0xffu];
        const uint2 mask  = t.sign_mask[(signs >> (4 * h)) & 15u];
        frag[h]           = make_uint2(entry.x ^ mask.x, entry.y ^ mask.y);
    }
}

// Contracts sub-block S of one 256-value block for MT row tiles and NT token tiles.
template <int S, int MT, int NT>
__device__ __forceinline__ void iq3xxs_sub_block_mma(const Iq3XxsTables& t,
                                                     const Iq3XxsLaneCodes (&c)[MT][2],
                                                     const __nv_bfloat16* const (&xrow)[NT],
                                                     const bool (&valid)[NT],
                                                     float (&acc)[MT][NT][4]) {
    uint4 xb[NT];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        xb[ni] = valid[ni] ? __ldg(reinterpret_cast<const uint4*>(xrow[ni] + S * kIq3XxsSubBlockK))
                           : make_uint4(0, 0, 0, 0);
    }
    uint2 f[MT][2][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        iq3xxs_lane_fragments<S>(t, c[mi][0], f[mi][0]);
        iq3xxs_lane_fragments<S>(t, c[mi][1], f[mi][1]);
    }
    float part[MT][NT][4] = {};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                mma_bf16(part[mi][ni][0], part[mi][ni][1], part[mi][ni][2], part[mi][ni][3],
                         f[mi][0][h].x, f[mi][1][h].x, f[mi][0][h].y, f[mi][1][h].y,
                         h == 0 ? xb[ni].x : xb[ni].z, h == 0 ? xb[ni].y : xb[ni].w);
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const float s0 =
            c[mi][0].d * (0.25f + 0.5f * static_cast<float>((c[mi][0].scales >> (4 * S)) & 15u));
        const float s8 =
            c[mi][1].d * (0.25f + 0.5f * static_cast<float>((c[mi][1].scales >> (4 * S)) & 15u));
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            acc[mi][ni][0] = fmaf(part[mi][ni][0], s0, acc[mi][ni][0]);
            acc[mi][ni][1] = fmaf(part[mi][ni][1], s0, acc[mi][ni][1]);
            acc[mi][ni][2] = fmaf(part[mi][ni][2], s8, acc[mi][ni][2]);
            acc[mi][ni][3] = fmaf(part[mi][ni][3], s8, acc[mi][ni][3]);
        }
    }
}

// Contracts one 256-value block of MT row tiles (rows gid and gid+8 of each) with NT token tiles.
template <int K, int MT, int NT>
__device__ __forceinline__ void iq3xxs_block_mma(const Iq3XxsTables& t,
                                                 const Iq3XxsLaneCodes (&c)[MT][2],
                                                 const __nv_bfloat16* __restrict__ x, int token0,
                                                 int tokens, int block, int gid, int lid,
                                                 float (&acc)[MT][NT][4]) {
    const __nv_bfloat16* xrow[NT];
    bool valid[NT];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int token = token0 + ni * 8 + gid;
        valid[ni]       = token < tokens;
        xrow[ni]        = x + static_cast<std::int64_t>(valid[ni] ? token : 0) * K +
                   block * kIq3XxsBlockK + lid * 8;
    }
    iq3xxs_sub_block_mma<0>(t, c, xrow, valid, acc);
    iq3xxs_sub_block_mma<1>(t, c, xrow, valid, acc);
    iq3xxs_sub_block_mma<2>(t, c, xrow, valid, acc);
    iq3xxs_sub_block_mma<3>(t, c, xrow, valid, acc);
    iq3xxs_sub_block_mma<4>(t, c, xrow, valid, acc);
    iq3xxs_sub_block_mma<5>(t, c, xrow, valid, acc);
    iq3xxs_sub_block_mma<6>(t, c, xrow, valid, acc);
    iq3xxs_sub_block_mma<7>(t, c, xrow, valid, acc);
}

// Cross-warp K reduction and epilogue. `partial` holds W * NT * 32 float4.
template <int W, int MT, int NT, bool Pair, class Out>
__device__ __forceinline__ void iq3xxs_reduce_store(float4* partial, const float (&acc)[MT][NT][4],
                                                    int warp, int lane, int token0, int tokens,
                                                    int row_tile0, const Out& out) {
    const int gid = lane >> 2;
    const int lid = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        __syncthreads();
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            partial[(warp * NT + ni) * 32 + lane] =
                make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
        }
        __syncthreads();
        for (int ni = warp; ni < NT; ni += W) {
            float c[4] = {};
#pragma unroll
            for (int w = 0; w < W; ++w) {
                const float4 p = partial[(w * NT + ni) * 32 + lane];
                c[0] += p.x;
                c[1] += p.y;
                c[2] += p.z;
                c[3] += p.w;
            }
            const int t = token0 + ni * 8 + lid * 2;
            if constexpr (Pair) {
                const int row    = (row_tile0 + mi) * 8 + gid;
                const auto store = [&](int token, float gate, float up) {
                    if (token < tokens) {
                        out.out[static_cast<std::int64_t>(token) * out.half_rows + row] =
                            __float2bfloat16_rn(silu(gate) * up);
                    }
                };
                store(t, c[0], c[2]);
                store(t + 1, c[1], c[3]);
            } else {
                const int row = (row_tile0 + mi) * 16 + gid;
                if (t < tokens) {
                    out.add(row, t, c[0]);
                    out.add(row + 8, t, c[2]);
                }
                if (t + 1 < tokens) {
                    out.add(row, t + 1, c[1]);
                    out.add(row + 8, t + 1, c[3]);
                }
            }
        }
    }
}

template <bool Pair, class Out>
__device__ __forceinline__ int iq3xxs_weight_row(const Out& out, int row_tile0, int local) {
    if constexpr (Pair) {
        return out.weight_row(row_tile0 * 8, local);
    } else {
        return row_tile0 * 16 + local;
    }
}

// Small-T kernel: CTA = MT row tiles x NT token tiles, W warps split the K blocks.
template <int K, int W, int MT, int NT, bool Pair, class Out>
__global__ __launch_bounds__(W * 32) void iq3xxs_mma_direct_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __half* __restrict__ d, int tokens, Out out) {
    constexpr int kBlocks   = K / kIq3XxsBlockK;
    constexpr int kRowBytes = kBlocks * kIq3XxsBlockCodeBytes;
    __shared__ Iq3XxsTables tables;
    __shared__ float4 partial[W * NT * 32];

    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int row_tile0 = static_cast<int>(blockIdx.x) * MT;
    const int token0    = static_cast<int>(blockIdx.y) * NT * 8;

    const std::uint8_t* row_codes[MT][2];
    const __half* row_d[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row    = iq3xxs_weight_row<Pair>(out, row_tile0, mi * 16 + gid + r * 8);
            row_codes[mi][r] = codes + static_cast<std::int64_t>(row) * kRowBytes;
            row_d[mi][r]     = d + static_cast<std::int64_t>(row) * kBlocks;
        }
    }
    struct Raw {
        uint4 index;
        uint2 signs;
        __half d;
    };
    const auto load = [&](int block, Raw (&raw)[MT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const std::uint8_t* base = row_codes[mi][r] + block * kIq3XxsBlockCodeBytes;
                raw[mi][r].index = __ldg(reinterpret_cast<const uint4*>(base + lid * 16));
                raw[mi][r].signs =
                    __ldg(reinterpret_cast<const uint2*>(base + kIq3XxsIndexBytes + lid * 8));
                raw[mi][r].d = __ldg(row_d[mi][r] + block);
            }
        }
    };

    tables.load();
    Raw cur[MT][2], next[MT][2];
    if (warp < kBlocks) load(warp, cur);
    __syncthreads();

    float acc[MT][NT][4] = {};
    for (int block = warp; block < kBlocks; block += W) {
        const bool more = block + W < kBlocks;
        if (more) load(block + W, next);
        Iq3XxsLaneCodes c[MT][2];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                // Gather the quad's eight sub-block scales (two per lane).
                const unsigned own = cur[mi][r].signs.y >> 24;
                unsigned scales    = 0;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    scales |= __shfl_sync(kFullWarpMask, own, (lane & ~3) | q) << (8 * q);
                }
                c[mi][r].index  = cur[mi][r].index;
                c[mi][r].signs  = (static_cast<std::uint64_t>(cur[mi][r].signs.y) << 32) |
                                 cur[mi][r].signs.x;
                c[mi][r].scales = scales;
                c[mi][r].d      = __half2float(cur[mi][r].d);
            }
        }
        iq3xxs_block_mma<K, MT, NT>(tables, c, x, token0, tokens, block, gid, lid, acc);
        if (more) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                cur[mi][0] = next[mi][0];
                cur[mi][1] = next[mi][1];
            }
        }
    }
    iq3xxs_reduce_store<W, MT, NT, Pair>(partial, acc, warp, lane, token0, tokens, row_tile0, out);
}

// Prefill kernel: cp.async stages the codes of 16*MT rows x W blocks per K tile.
template <int K, int W, int MT, int NT, int S, bool Pair, class Out>
__global__ __launch_bounds__(W * 32) void iq3xxs_mma_staged_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __half* __restrict__ d, int tokens, Out out) {
    constexpr int kBlocks   = K / kIq3XxsBlockK;
    constexpr int kRowBytes = kBlocks * kIq3XxsBlockCodeBytes;
    constexpr int kTiles    = (kBlocks + W - 1) / W;
    constexpr int kChunks   = kIq3XxsBlockCodeBytes / 16;
    constexpr int kRows     = 16 * MT;

    union SharedStorage {
        alignas(16) std::uint8_t codes[S][kRows][W][kIq3XxsBlockCodeBytes];
        float4 partial[W * NT * 32];
    };
    __shared__ SharedStorage shared;
    __shared__ Iq3XxsTables tables;

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int row_tile0 = static_cast<int>(blockIdx.x) * MT;
    const int token0    = static_cast<int>(blockIdx.y) * NT * 8;

    const auto stage = [&](int slot, int tile) {
        for (int item = tid; item < kRows * W * kChunks; item += W * 32) {
            const int chunk  = item % kChunks;
            const int w      = (item / kChunks) % W;
            const int row    = item / (kChunks * W);
            const int block  = tile * W + w;
            const bool valid = block < kBlocks;
            const std::uint8_t* src =
                codes +
                static_cast<std::int64_t>(iq3xxs_weight_row<Pair>(out, row_tile0, row)) *
                    kRowBytes +
                (valid ? block : 0) * kIq3XxsBlockCodeBytes + chunk * 16;
            cp_async_zfill<16, Cache::cg>(&shared.codes[slot][row][w][chunk * 16], src,
                                          valid ? 16 : 0);
        }
        cp_commit();
    };

    tables.load();
#pragma unroll
    for (int slot = 0; slot < S; ++slot) {
        if (slot < kTiles) {
            stage(slot, slot);
        } else {
            cp_commit();
        }
    }
    const __half* row_d[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = iq3xxs_weight_row<Pair>(out, row_tile0, mi * 16 + gid + r * 8);
            row_d[mi][r]  = d + static_cast<std::int64_t>(row) * kBlocks;
        }
    }

    float acc[MT][NT][4] = {};
#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int slot  = tile % S;
        const int block = tile * W + warp;
        cp_wait<S - 1>();
        __syncthreads();
        if (block < kBlocks) {
            Iq3XxsLaneCodes c[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const std::uint8_t* b = shared.codes[slot][mi * 16 + gid + r * 8][warp];
                    const uint2 signs =
                        *reinterpret_cast<const uint2*>(b + kIq3XxsIndexBytes + lid * 8);
                    c[mi][r].index  = *reinterpret_cast<const uint4*>(b + lid * 16);
                    c[mi][r].signs  = (static_cast<std::uint64_t>(signs.y) << 32) | signs.x;
                    c[mi][r].scales = static_cast<unsigned>(b[71]) |
                                      (static_cast<unsigned>(b[79]) << 8) |
                                      (static_cast<unsigned>(b[87]) << 16) |
                                      (static_cast<unsigned>(b[95]) << 24);
                    c[mi][r].d = __half2float(__ldg(row_d[mi][r] + block));
                }
            }
            iq3xxs_block_mma<K, MT, NT>(tables, c, x, token0, tokens, block, gid, lid, acc);
        }
        if (tile + S < kTiles) {
            __syncthreads();
            stage(slot, tile + S);
        } else {
            cp_commit();
        }
    }
    cp_wait<0>();
    iq3xxs_reduce_store<W, MT, NT, Pair>(shared.partial, acc, warp, lane, token0, tokens,
                                         row_tile0, out);
}

// ---------------------------------------------------------------------------------------------
// Int8 prefill path. Activations are quantized per token and group of 32*SG values (absmax /
// 127); signed grid magnitudes (|v| <= 62) are exact int8, so mma.m16n8k32.s8 contracts one
// sub-block per instruction at 4x the BF16 rate. m16n8k32 gives lane quad position q the
// k-positions [4q, 4q+4) and [16+4q, 16+4q+4) of rows gid and gid+8, i.e. exactly grid entries 2q
// and 2q+1 as iq3_xxs_planar_v1 already stores them, and activations 8q..8q+7 in natural order.
//
// The sub-block scale d*(0.25 + 0.5*ls) is (d/4) * (1 + 2*ls) with an integer factor, so the
// partials of one activation group stay in int32 (one IMAD per element and sub-block; |sum| <=
// 62*127*32*31*8 < 2^31) and are converted and scaled once per group. Per-sub-block FP32
// rescaling would cost three FP32 instructions per element and saturates issue before the tensor
// pipe does.

// Token rows of the staged activation tile are 256 bytes with 16-byte chunk c of token t stored at
// chunk c ^ ((t & 7) << 1), so the B-fragment LDS.64 of a quad-row hits 32 distinct banks.
constexpr int kIq3XxsQ8XRowBytes = 256;
constexpr int kIq3XxsQ8Group     = kQ8ActGroup / kIq3XxsSubBlockK; // sub-blocks per scale

// Signed int8x4 value of every (grid index, 4-bit sign nibble): table[index * 16 + nibble].
__device__ __forceinline__ void iq3xxs_load_signed_grid_s8(std::uint32_t* table) {
    for (int i = static_cast<int>(threadIdx.x); i < 4096; i += static_cast<int>(blockDim.x)) {
        const unsigned nib = static_cast<unsigned>(i) & 15u;
        // 0xff in every byte whose sign bit is set; magnitudes <= 62 so ~m + 1 never carries.
        const unsigned m = ((nib & 1u) * 0xffu) | ((nib & 2u) * 0x7f80u) |
                           ((nib & 4u) * 0x3fc000u) | ((nib & 8u) * 0x1fe00000u);
        table[i] = (kIq3XxsGrid[i >> 4] ^ m) + (m & 0x01010101u);
    }
}

template <int BN, int SG>
struct Iq3XxsQ8Smem {
    static constexpr int kTable  = 4096 * 4;
    static constexpr int kX      = BN * kIq3XxsQ8XRowBytes;
    static constexpr int kScales = (8 / SG) * BN * 4;
    static constexpr int kStage  = kX + kScales;
    static constexpr int kStages = 2;
    static constexpr int kBytes  = kTable + kStages * kStage;
    static_assert(kX % 16 == 0, "stage planes must stay 16-byte aligned");
};

// Prefill GEMM: CTA = WM warps x (16*MT rows) over BN = 8*NT tokens and the full K. Each warp
// owns its rows and loads their codes straight into registers one block ahead (no other warp
// reads them); the token tile's int8 activations and scales are shared through a 2-stage cp.async
// ring. grid = (ceil(tokens / BN), rows / (16*MT*WM)); token tiles vary fastest so a weight tile
// is re-read from L2 by neighbouring CTAs.
template <int K, int MT, int NT, int WM, int SG, int MB, bool Pair, class Out>
__global__ __launch_bounds__(WM * 32, MB) void iq3xxs_q8_mma_kernel(
    const std::int8_t* __restrict__ xq, const float* __restrict__ sx, int sx_stride,
    const std::uint8_t* __restrict__ codes, const __half* __restrict__ d, int tokens, Out out) {
    constexpr int kBlocks   = K / kIq3XxsBlockK;
    constexpr int kRowBytes = kBlocks * kIq3XxsBlockCodeBytes;
    constexpr int kThreads  = WM * 32;
    constexpr int BM        = 16 * MT * WM;
    constexpr int BN        = 8 * NT;
    constexpr int kGroups   = 8 / SG; // activation scales per block
    using L                 = Iq3XxsQ8Smem<BN, SG>;
    static_assert(kQ8ActTokenPad % BN == 0, "sx padding must cover the token tile");

    extern __shared__ __align__(16) std::uint8_t smem[];
    auto* table = reinterpret_cast<std::uint32_t*>(smem);

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int token0 = static_cast<int>(blockIdx.x) * BN;
    const int tile0  = static_cast<int>(blockIdx.y) * (BM / 16) + warp * MT; // warp's first tile
    const auto stage_base = [&](int slot) { return smem + L::kTable + slot * L::kStage; };

    const auto stage = [&](int slot, int block) {
        std::uint8_t* xs = stage_base(slot);
        for (int i = tid; i < BN * 16; i += kThreads) {
            const int t = i >> 4, chunk = i & 15;
            const bool valid = token0 + t < tokens;
            const std::int8_t* src = xq +
                                     static_cast<std::int64_t>(valid ? token0 + t : 0) * K +
                                     block * kIq3XxsBlockK + chunk * 16;
            cp_async_zfill<16, Cache::cg>(
                xs + t * kIq3XxsQ8XRowBytes + ((chunk ^ ((t & 7) << 1)) * 16), src, valid ? 16 : 0);
        }
        std::uint8_t* ss = xs + L::kX;
        for (int i = tid; i < kGroups * (BN / 4); i += kThreads) {
            const int g = i / (BN / 4), chunk = i % (BN / 4);
            const float* src = sx + static_cast<std::int64_t>(block * kGroups + g) * sx_stride +
                               token0 + chunk * 4;
            cp_async_zfill<16, Cache::cg>(ss + (g * BN + chunk * 4) * 4, src, 16);
        }
        cp_commit();
    };

    const std::uint8_t* row_codes[MT][2];
    const __half* row_d[MT][2];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row    = iq3xxs_weight_row<Pair>(out, tile0, mi * 16 + gid + r * 8);
            row_codes[mi][r] = codes + static_cast<std::int64_t>(row) * kRowBytes;
            row_d[mi][r]     = d + static_cast<std::int64_t>(row) * kBlocks;
        }
    }
    struct Raw {
        uint4 index;
        uint2 signs;
        __half d;
    };
    const auto load = [&](int block, Raw (&raw)[MT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const std::uint8_t* base = row_codes[mi][r] + block * kIq3XxsBlockCodeBytes;
                raw[mi][r].index = __ldg(reinterpret_cast<const uint4*>(base + lid * 16));
                raw[mi][r].signs =
                    __ldg(reinterpret_cast<const uint2*>(base + kIq3XxsIndexBytes + lid * 8));
                raw[mi][r].d = __ldg(row_d[mi][r] + block);
            }
        }
    };

    // Weight codes are register-prefetched two blocks ahead: they come from L2/DRAM with no reuse
    // inside the CTA, and one block of lead does not cover their latency.
    stage(0, 0);
    Raw cur[MT][2], next[MT][2], after[MT][2];
    load(0, cur);
    if (kBlocks > 1) load(1, next);
    iq3xxs_load_signed_grid_s8(table);

    float acc[MT][NT][4] = {};
#pragma unroll 1
    for (int block = 0; block < kBlocks; ++block) {
        const bool more = block + 1 < kBlocks;
        if (block + 2 < kBlocks) load(block + 2, after);
        if (more) {
            stage((block + 1) & 1, block + 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        Iq3XxsLaneCodes c[MT][2];
        float dq[MT][2]; // d / 4
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                // Gather the quad's eight sub-block scales (two per lane).
                const unsigned own = cur[mi][r].signs.y >> 24;
                unsigned scales    = 0;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    scales |= __shfl_sync(kFullWarpMask, own, (lane & ~3) | q) << (8 * q);
                }
                c[mi][r].index  = cur[mi][r].index;
                c[mi][r].signs  = (static_cast<std::uint64_t>(cur[mi][r].signs.y) << 32) |
                                 cur[mi][r].signs.x;
                c[mi][r].scales = scales;
                dq[mi][r]       = 0.25f * __half2float(cur[mi][r].d);
            }
        }
        const std::uint8_t* xs   = stage_base(block & 1);
        const float* ss          = reinterpret_cast<const float*>(xs + L::kX);
        const std::uint8_t* xrow = xs + gid * kIq3XxsQ8XRowBytes + (lid >> 1) * 16 + (lid & 1) * 8;
        const float* srow        = ss + lid * 2;
        int iacc[MT][NT][4];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            unsigned a[MT][4];
            int f[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const unsigned word = s < 2   ? c[mi][r].index.x
                                          : s < 4 ? c[mi][r].index.y
                                          : s < 6 ? c[mi][r].index.z
                                                  : c[mi][r].index.w;
                    const unsigned pair = word >> (16 * (s & 1));
                    const unsigned sg =
                        iq3xxs_sign8(static_cast<unsigned>(c[mi][r].signs >> (7 * s)) & 127u);
                    a[mi][r]     = table[((pair & 0xffu) << 4) | (sg & 15u)];
                    a[mi][r + 2] = table[((pair >> 4) & 0xff0u) | (sg >> 4)];
                    f[mi][r]     = 1 + 2 * static_cast<int>((c[mi][r].scales >> (4 * s)) & 15u);
                }
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                // Sub-block s is chunks 2s and 2s+1 of the row, swizzled by the row's low bits (gid).
                const uint2 xb = *reinterpret_cast<const uint2*>(xrow + ni * 8 * kIq3XxsQ8XRowBytes +
                                                               ((s ^ gid) << 5));
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                    mma_s8(p0, p1, p2, p3, a[mi][0], a[mi][1], a[mi][2], a[mi][3], xb.x, xb.y);
                    if (s % SG == 0) {
                        iacc[mi][ni][0] = p0 * f[mi][0];
                        iacc[mi][ni][1] = p1 * f[mi][0];
                        iacc[mi][ni][2] = p2 * f[mi][1];
                        iacc[mi][ni][3] = p3 * f[mi][1];
                    } else {
                        iacc[mi][ni][0] += p0 * f[mi][0];
                        iacc[mi][ni][1] += p1 * f[mi][0];
                        iacc[mi][ni][2] += p2 * f[mi][1];
                        iacc[mi][ni][3] += p3 * f[mi][1];
                    }
                }
            }
            if (s % SG == SG - 1) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const float2 xs2 = *reinterpret_cast<const float2*>(srow + (s / SG) * BN + ni * 8);
#pragma unroll
                    for (int mi = 0; mi < MT; ++mi) {
                        acc[mi][ni][0] = fmaf(static_cast<float>(iacc[mi][ni][0]), dq[mi][0] * xs2.x, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(static_cast<float>(iacc[mi][ni][1]), dq[mi][0] * xs2.y, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(static_cast<float>(iacc[mi][ni][2]), dq[mi][1] * xs2.x, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(static_cast<float>(iacc[mi][ni][3]), dq[mi][1] * xs2.y, acc[mi][ni][3]);
                    }
                }
            }
        }
        if (more) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                cur[mi][0]  = next[mi][0];
                cur[mi][1]  = next[mi][1];
                next[mi][0] = after[mi][0];
                next[mi][1] = after[mi][1];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int tile = tile0 + mi;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int t = token0 + ni * 8 + lid * 2;
            const float* v = acc[mi][ni];
            if constexpr (Pair) {
                const int row = tile * 8 + gid;
                if (t < tokens) {
                    out.out[static_cast<std::int64_t>(t) * out.half_rows + row] =
                        __float2bfloat16_rn(silu(v[0]) * v[2]);
                }
                if (t + 1 < tokens) {
                    out.out[static_cast<std::int64_t>(t + 1) * out.half_rows + row] =
                        __float2bfloat16_rn(silu(v[1]) * v[3]);
                }
            } else {
                const int row = tile * 16 + gid;
                if (t < tokens) {
                    out.add(row, t, v[0]);
                    out.add(row + 8, t, v[2]);
                }
                if (t + 1 < tokens) {
                    out.add(row, t + 1, v[1]);
                    out.add(row + 8, t + 1, v[3]);
                }
            }
        }
    }
}

// Host side of the int8 path: activations come from q8_quantize_act<K, Q8ActLayout::Natural>.
constexpr int kIq3XxsQ8MinTokens     = 33;
constexpr int kIq3XxsQ8WideMinTokens = 1024;

template <int K, int MT, int NT, int WM, bool Pair, int MB = 1, int SG = kIq3XxsQ8Group, class Out>
void iq3xxs_launch_q8(const Q8Act& act, const std::uint8_t* codes, const __half* d,
                      int rows, int tokens, const Out& out, cudaStream_t stream) {
    constexpr int BM    = 16 * MT * WM;
    constexpr int BN    = 8 * NT;
    constexpr int kSmem = Iq3XxsQ8Smem<BN, SG>::kBytes;
    auto kernel         = iq3xxs_q8_mma_kernel<K, MT, NT, WM, SG, MB, Pair, Out>;
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        return true;
    }();
    (void)configured;
    if (rows % BM != 0) throw std::invalid_argument("iq3xxs int8 GEMM: rows must tile by BM");
    const dim3 grid((tokens + BN - 1) / BN, rows / BM);
    kernel<<<grid, WM * 32, kSmem, stream>>>(act.xq, act.sx, act.sx_stride, codes, d, tokens, out);
}

} // namespace ninfer::ops::detail
