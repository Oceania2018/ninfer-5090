#include "ops/linear_add/iq3_xxs/iq3_xxs_linear_add.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/iq3xxs_matmul.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120;
constexpr int kK    = 17408;

template <int W, int MT, int NT>
void launch_direct(const __nv_bfloat16* x, const std::uint8_t* codes, const __half* d, int tokens,
                   Iq3XxsResidualOut out, cudaStream_t stream) {
    static_assert(kRows % (16 * MT) == 0, "row tiles must cover the rows exactly");
    const dim3 grid(kRows / (16 * MT), (tokens + NT * 8 - 1) / (NT * 8));
    iq3xxs_mma_direct_kernel<kK, W, MT, NT, false>
        <<<grid, W * 32, 0, stream>>>(x, codes, d, tokens, out);
}

template <int W, int MT, int NT, int S>
void launch_staged(const __nv_bfloat16* x, const std::uint8_t* codes, const __half* d, int tokens,
                   Iq3XxsResidualOut out, cudaStream_t stream) {
    static_assert(kRows % (16 * MT) == 0, "row tiles must cover the rows exactly");
    const dim3 grid(kRows / (16 * MT), (tokens + NT * 8 - 1) / (NT * 8));
    iq3xxs_mma_staged_kernel<kK, W, MT, NT, S, false>
        <<<grid, W * 32, 0, stream>>>(x, codes, d, tokens, out);
}

} // namespace

std::size_t iq3_xxs_linear_add_workspace_bytes(std::int32_t max_tokens) noexcept {
    return max_tokens < kIq3XxsQ8MinTokens ? 0 : q8_act_workspace_bytes(kK, max_tokens);
}

bool iq3_xxs_linear_add_admits(std::int32_t output_rows, std::int32_t input_rows) noexcept {
    return output_rows == kRows && input_rows == kK;
}

void iq3_xxs_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                 bool int8_activations, WorkspaceArena& ws,
                                 cudaStream_t stream) {
    if (w.qtype != QType::IQ3_XXS || w.layout != QuantLayout::Iq3XxsPlanar || w.n != kRows ||
        w.k != kK || w.padded_shape[1] != kK || w.qdata == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("iq3_xxs linear_add: weight must be [5120,17408] IQ3_XXS");
    }
    const auto* xp    = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes = static_cast<const std::uint8_t*>(w.qdata);
    const auto* d     = static_cast<const __half*>(w.scales);
    const Iq3XxsResidualOut o{static_cast<__nv_bfloat16*>(residual_out.data),
                              static_cast<std::int64_t>(residual_out.nb[1] / sizeof(__nv_bfloat16))};
    const int tokens = x.ne[1];
#if defined(NINFER_SM86)
    // sm_89 (RTX 4090), measured: T=1 48.4 -> 44.4 us with 8 warps; T=32 107.2 -> 87.8 us with 2.
    if (tokens <= 8) {
        launch_direct<8, 1, 1>(xp, codes, d, tokens, o, stream);
    } else if (tokens <= 16) {
        launch_direct<4, 1, 2>(xp, codes, d, tokens, o, stream);
    } else if (tokens <= 32) {
        launch_direct<2, 1, 4>(xp, codes, d, tokens, o, stream);
    }
#else
    if (tokens <= 8) {
        launch_direct<4, 1, 1>(xp, codes, d, tokens, o, stream);
    } else if (tokens <= 16) {
        launch_direct<4, 1, 2>(xp, codes, d, tokens, o, stream);
    } else if (tokens <= 32) {
        launch_direct<4, 1, 4>(xp, codes, d, tokens, o, stream);
    }
#endif
    else if (!int8_activations) {
        launch_staged<4, 2, 8, 2>(xp, codes, d, tokens, o, stream);
    } else {
        auto scope     = ws.scope();
        const auto act = q8_quantize_act<kK, Q8ActLayout::Natural>(xp, tokens, ws, stream);
        // 4 warps x 16 rows over 64 tokens, two CTAs per SM (measured best of the 16..256-row tiles).
        // Full prefill chunks: 128-token tiles amortize each decoded weight fragment over twice
        // the tokens (+5% at T >= 1024); smaller T keeps 64-token tiles for enough CTAs.
        if (tokens >= kIq3XxsQ8WideMinTokens) {
            iq3xxs_launch_q8<kK, 1, 16, 8, false, 1>(act, codes, d, kRows, tokens, o, stream);
        } else {
            iq3xxs_launch_q8<kK, 1, 8, 4, false, 2>(act, codes, d, kRows, tokens, o, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
