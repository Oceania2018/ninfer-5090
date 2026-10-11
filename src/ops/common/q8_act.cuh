#pragma once

// Int8 activations for the int8 tensor-core prefill paths (IQ3_XXS and Q4/Q5 weights).
//
// x [tokens, K] BF16 is quantized per token and 256-value group with absmax / 127 scales. The
// codes go to xq [tokens, K] int8 in the byte order the consuming GEMM loads as B fragments:
//
//   Natural  - unchanged (IQ3_XXS planar kernels permute the weights instead).
//   Q4Q5Lane - inside every 64-value group, quad lane q's 16 bytes are values 16q + {0,2,4,6},
//              16q + {1,3,5,7}, 16q + {8,10,12,14}, 16q + {9,11,13,15}: the b0/b1 operands of the
//              group's two m16n8k32 steps, matching the even/odd code nibbles a lane owns.
//
// Scales go to sx [K/256, sx_stride] FP32, group-major so a GEMM stages one group of a token tile
// with 16-byte copies; sx_stride pads the token count to kQ8ActTokenPad.

#include "core/arena.h"
#include "ops/common/q8_act.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8ActLayout : std::uint8_t { Natural, Q4Q5Lane };

struct Q8Act {
    const std::int8_t* xq;
    const float* sx;
    int sx_stride;
};

// 8 values per thread, 32 threads (one warp) per 256-value group.
template <int K, Q8ActLayout Layout>
__global__ __launch_bounds__(256) void q8_quantize_act_kernel(const __nv_bfloat16* __restrict__ x,
                                                              std::int8_t* __restrict__ xq,
                                                              float* __restrict__ sx, int tokens,
                                                              int sx_stride) {
    static_assert(K % kQ8ActGroup == 0, "whole warps must stay inside one token");
    const int item  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int token = item / (K / 8);
    if (token >= tokens) return; // uniform per warp
    const int chunk = item % (K / 8);
    const uint4 raw =
        __ldg(reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * K) + chunk);
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    float v[8];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i]     = __uint_as_float(words[i] << 16);
        v[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
        amax         = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) amax = fmaxf(amax, __shfl_xor_sync(kFullWarpMask, amax, m));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    unsigned q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = static_cast<unsigned>(__float2int_rn(v[i] * inv)) & 0xffu;
    uint2 packed;
    if constexpr (Layout == Q8ActLayout::Natural) {
        packed = make_uint2(q[0] | q[1] << 8 | q[2] << 16 | q[3] << 24,
                            q[4] | q[5] << 8 | q[6] << 16 | q[7] << 24);
    } else {
        // The 8 values are one half (j0 = 0 or 8) of a lane's 16: evens first, then odds. The
        // half stays at its own 8-byte position, so the store offset is unchanged.
        packed = make_uint2(q[0] | q[2] << 8 | q[4] << 16 | q[6] << 24,
                            q[1] | q[3] << 8 | q[5] << 16 | q[7] << 24);
    }
    *reinterpret_cast<uint2*>(xq + static_cast<std::int64_t>(token) * K + chunk * 8) = packed;
    if ((chunk & 31) == 0) {
        sx[static_cast<std::int64_t>(chunk / 32) * sx_stride + token] = amax / 127.0f;
    }
}

// Allocates from `ws` (the caller owns the scope) and quantizes x [tokens, K].
template <int K, Q8ActLayout Layout>
Q8Act q8_quantize_act(const __nv_bfloat16* x, int tokens, WorkspaceArena& ws, cudaStream_t stream) {
    const int stride = q8_act_sx_stride(tokens);
    auto* xq = static_cast<std::int8_t*>(ws.alloc_bytes(static_cast<std::size_t>(tokens) * K).data);
    auto* sx = static_cast<float*>(
        ws.alloc_bytes(static_cast<std::size_t>(K / kQ8ActGroup) * stride * 4).data);
    const std::int64_t threads = static_cast<std::int64_t>(tokens) * (K / 8);
    q8_quantize_act_kernel<K, Layout><<<static_cast<unsigned>((threads + 255) / 256), 256, 0, stream>>>(
        x, xq, sx, tokens, stride);
    return {xq, sx, stride};
}

} // namespace ninfer::ops::detail
