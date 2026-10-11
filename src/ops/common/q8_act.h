#pragma once

// Host-side sizing for the int8 activation paths (see q8_act.cuh).

#include <cstddef>

namespace ninfer::ops::detail {

constexpr int kQ8ActGroup    = 256; // values per activation scale
constexpr int kQ8ActTokenPad = 128; // sx token padding (largest GEMM token tile)
constexpr int kQ4Q5Q8MinTokens = 64; // Q4/Q5 int8 route from this many tokens

inline int q8_act_sx_stride(int tokens) {
    return (tokens + kQ8ActTokenPad - 1) / kQ8ActTokenPad * kQ8ActTokenPad;
}

// Workspace for q8_quantize_act at up to max_tokens (two 256-aligned allocations).
inline std::size_t q8_act_workspace_bytes(int k, int max_tokens) {
    const auto align = [](std::size_t v) { return (v + 255) / 256 * 256; };
    return align(static_cast<std::size_t>(max_tokens) * k) +
           align(static_cast<std::size_t>(k / kQ8ActGroup) * q8_act_sx_stride(max_tokens) * 4) +
           256;
}

} // namespace ninfer::ops::detail
