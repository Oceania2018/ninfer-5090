// Q4/Q5 int8-activation prefill routes (LinearPolicy::AllowA8, T >= 64): paired GDN input,
// paired attention input and the Q5 [5120, 6144] linear_add. Each output is checked tightly
// against an FP64 oracle fed the activations the kernels see (per-256 absmax int8, mirroring
// q8_quantize_act's FP32 arithmetic), and its distance from the exact-BF16 result is reported.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr ReductionCriterion kQuantAwareTolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kVsBf16Tolerance{0.02, 1.0, 0.05};

// Activations with a few large channels, as real hidden states have.
std::vector<float> make_activation(std::int32_t hidden, std::int32_t tokens, std::uint32_t seed) {
    std::vector<float> values = make_bf16_activation(hidden, tokens, seed);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 37; c < hidden; c += 997) {
            values[static_cast<std::size_t>(t) * hidden + c] *= 40.0F;
        }
    }
    round_to_bf16(values);
    return values;
}

std::vector<float> int8_activation(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (std::size_t g = 0; g < x.size(); g += 256) {
        float amax = 0.0F;
        for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(x[g + i]));
        const float inv = amax > 0.0F ? 127.0F / amax : 0.0F, scale = amax / 127.0F;
        for (int i = 0; i < 256; ++i) {
            out[g + i] = static_cast<float>(std::nearbyint(x[g + i] * inv)) * scale;
        }
    }
    return out;
}

int check(const std::string& label, const GuardedBf16Tensor& out, std::int32_t full_rows,
          std::int32_t out_offset, std::int32_t rows, const quantized_weight::PackedWeight& w,
          std::int32_t w_offset, const std::vector<float>& xq, const std::vector<float>& x,
          std::int32_t hidden, std::int32_t tokens, const std::vector<double>* base = nullptr) {
    std::vector<double> actual = gather_rows(out.values(), full_rows, out_offset, rows, tokens);
    if (base != nullptr) { // linear_add: subtract the residual the output started from
        const std::vector<double> b = gather_rows(*base, full_rows, out_offset, rows, tokens);
        for (std::size_t i = 0; i < actual.size(); ++i) actual[i] -= b[i];
    }
    int failures = compare(label + " vs int8 oracle", actual,
                           projection_oracle(w, w_offset, rows, xq, hidden, tokens),
                           kQuantAwareTolerance);
    failures += compare(label + " vs BF16 oracle", actual,
                        projection_oracle(w, w_offset, rows, x, hidden, tokens), kVsBf16Tolerance);
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr std::int32_t kHidden = 5120;
    quantized_weight::PatternedWeightOptions options;
    options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
    options.row_split_scale = quantized_weight::RowSplitScalePattern::Small;
    DevicePackedWeight gdn_qk(
        quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 4096, kHidden, 409U, options));
    DevicePackedWeight gdn_vz(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 12288, kHidden, 419U, options));
    DevicePackedWeight attn_qk(
        quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 7168, kHidden, 103U, options));
    DevicePackedWeight attn_gv(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 7168, kHidden, 107U, options));
    DevicePackedWeight out_proj(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 5120, 6144, 211U, options));
    DeviceArena ws(std::size_t{256} << 20);

    int failures = 0;
    for (const std::int32_t tokens : {64, 65, 100, 128, 257}) {
        const std::string suffix = " T=" + std::to_string(tokens);
        const std::vector<float> x  = make_activation(kHidden, tokens, 401U + tokens);
        const std::vector<float> xq = int8_activation(x);
        const std::vector<std::uint16_t> x_bits = bf16_bits(x);
        DeviceBuffer dx = to_device(x_bits);
        Tensor xt(dx.p, DType::BF16, {kHidden, tokens});

        // GDN input: qk (Q4) -> qkv[0, 4096), value (Q5 rows 0..6143) -> qkv[4096, 10240),
        // z (Q5 rows 6144..12287) -> z.
        GuardedBf16Tensor qkv(10240, tokens), z(6144, tokens);
        Tensor qkv_t = qkv.tensor(), z_t = z.tensor();
        ops::gdn_input_proj(xt, gdn_qk.view(), gdn_vz.view(), qkv_t, z_t,
                            ops::LinearPolicy::AllowA8, ws, nullptr);
        cuda_synchronize();
        failures += qkv.verify_guards("gdn qkv" + suffix) + z.verify_guards("gdn z" + suffix);
        failures += check("gdn qk" + suffix, qkv, 10240, 0, 4096, gdn_qk.host, 0, xq, x, kHidden,
                          tokens);
        failures += check("gdn value" + suffix, qkv, 10240, 4096, 6144, gdn_vz.host, 0, xq, x,
                          kHidden, tokens);
        failures += check("gdn z" + suffix, z, 6144, 0, 6144, gdn_vz.host, 6144, xq, x, kHidden,
                          tokens);

        // Attention input: q/k from the Q4 parent, gate/v from the Q5 parent.
        GuardedBf16Tensor q(6144, tokens), g(6144, tokens), k(1024, tokens), v(1024, tokens);
        Tensor q_t = q.tensor(), g_t = g.tensor(), k_t = k.tensor(), v_t = v.tensor();
        ops::attn_input_proj(xt, attn_qk.view(), attn_gv.view(), q_t, g_t, k_t, v_t,
                             ops::LinearPolicy::AllowA8, ws, nullptr);
        cuda_synchronize();
        failures += check("attn q" + suffix, q, 6144, 0, 6144, attn_qk.host, 0, xq, x, kHidden,
                          tokens);
        failures += check("attn k" + suffix, k, 1024, 0, 1024, attn_qk.host, 6144, xq, x, kHidden,
                          tokens);
        failures += check("attn gate" + suffix, g, 6144, 0, 6144, attn_gv.host, 0, xq, x, kHidden,
                          tokens);
        failures += check("attn v" + suffix, v, 1024, 0, 1024, attn_gv.host, 6144, xq, x, kHidden,
                          tokens);

        // Output projection: residual += W x with K = 6144.
        const std::vector<float> xo  = make_activation(6144, tokens, 901U + tokens);
        const std::vector<float> xoq = int8_activation(xo);
        const std::vector<std::uint16_t> xo_bits = bf16_bits(xo);
        DeviceBuffer dxo = to_device(xo_bits);
        Tensor xo_t(dxo.p, DType::BF16, {6144, tokens});
        std::vector<float> residual = make_bf16_activation(5120, tokens, 77U + tokens);
        const std::vector<std::uint16_t> residual_bits = bf16_bits(residual);
        GuardedBf16Tensor res(5120, tokens);
        res.copy_from_bits(residual_bits);
        const std::vector<double> residual_before = res.values();
        Tensor res_t = res.tensor();
        ops::linear_add(xo_t, out_proj.view(), res_t, ops::LinearPolicy::AllowA8, ws, nullptr);
        cuda_synchronize();
        failures += check("linear_add" + suffix, res, 5120, 0, 5120, out_proj.host, 0, xoq, xo,
                          6144, tokens, &residual_before);
        std::cout << "T=" << tokens << (failures ? " FAILED so far" : " ok") << "\n";
    }
    std::cout << (failures ? "FAILED" : "PASSED") << " (" << failures << ")\n";

    // Optional throughput probe: NINFER_Q8PROJ_BENCH=1 (weights stay L2-resident; compare relatively).
    if (const char* bench = std::getenv("NINFER_Q8PROJ_BENCH"); bench && *bench == '1') {
        constexpr std::int32_t kTokens = 1024;
        const std::vector<std::uint16_t> xb = bf16_bits(make_activation(6144, kTokens, 5U));
        DeviceBuffer dx = to_device(xb);
        Tensor x5(dx.p, DType::BF16, {kHidden, kTokens}), x6(dx.p, DType::BF16, {6144, kTokens});
        GuardedBf16Tensor qkv(10240, kTokens), z(6144, kTokens), q(6144, kTokens), g(6144, kTokens),
            k(1024, kTokens), v(1024, kTokens), res(5120, kTokens);
        Tensor qkv_t = qkv.tensor(), z_t = z.tensor(), q_t = q.tensor(), g_t = g.tensor(),
               k_t = k.tensor(), v_t = v.tensor(), res_t = res.tensor();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        const auto time_us = [&](const char* name, double rows, double k_dim, auto&& launch) {
            for (int i = 0; i < 5; ++i) launch();
            cudaEventRecord(start);
            for (int i = 0; i < 50; ++i) launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            const double us = ms * 1e3 / 50;
            std::printf("bench %-10s T=%d %8.1f us %6.0f TOPS\n", name, kTokens, us,
                        2.0 * rows * k_dim * kTokens / (us * 1e6));
        };
        time_us("gdn_in", 16384, kHidden, [&] {
            ops::gdn_input_proj(x5, gdn_qk.view(), gdn_vz.view(), qkv_t, z_t,
                                ops::LinearPolicy::AllowA8, ws, nullptr);
        });
        time_us("attn_in", 14336, kHidden, [&] {
            ops::attn_input_proj(x5, attn_qk.view(), attn_gv.view(), q_t, g_t, k_t, v_t,
                                 ops::LinearPolicy::AllowA8, ws, nullptr);
        });
        time_us("out_proj", 5120, 6144, [&] {
            ops::linear_add(x6, out_proj.view(), res_t, ops::LinearPolicy::AllowA8, ws, nullptr);
        });
    }
    return failures ? 1 : 0;
}
