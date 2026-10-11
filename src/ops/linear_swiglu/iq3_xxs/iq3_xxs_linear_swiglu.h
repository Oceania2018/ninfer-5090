#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Qwen3.8-27B dense MLP only: gate rows [0, 17408) and up rows [17408, 34816) of one
// [34816, 5120] IQ3_XXS planar object.
// With int8_activations (LinearPolicy allows A8), T > 32 quantizes activations to int8 in the
// workspace (int8 tensor-core prefill); otherwise BF16 activations for every T.
bool iq3_xxs_linear_swiglu_admits(std::int32_t gate_up_rows, std::int32_t input_rows) noexcept;

std::size_t iq3_xxs_linear_swiglu_workspace_bytes(std::int32_t max_tokens) noexcept;

void iq3_xxs_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                                    bool int8_activations, WorkspaceArena& ws,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
