#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Qwen3.8-27B dense MLP down projection only: [5120, 17408] IQ3_XXS planar weight,
// residual_out += W x.
// With int8_activations (LinearPolicy allows A8), T > 32 quantizes activations to int8 in the
// workspace (int8 tensor-core prefill); otherwise BF16 activations for every T.
bool iq3_xxs_linear_add_admits(std::int32_t output_rows, std::int32_t input_rows) noexcept;

std::size_t iq3_xxs_linear_add_workspace_bytes(std::int32_t max_tokens) noexcept;

void iq3_xxs_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                 bool int8_activations, WorkspaceArena& ws,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
