// IQ3_XXS dense MLP Ops (LinearSwiGLU gate/up and LinearAdd down) against an FP64 host oracle.
// Random llama.cpp-style blocks exercise every grid index, sign pattern and sub-block scale; the
// oracle decodes them with the reference dequantize_row_iq3_xxs semantics.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;

// llama.cpp ggml-common.h iq3xxs_grid (MIT); duplicated so the oracle is independent of the kernel.
#include "iq3_xxs_grid.inc"

std::uint16_t bf16_bits(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<std::uint16_t>(u >> 16);
}

float bf16_value(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

float half_value(std::uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const float s = (h & 0x8000) ? -1.0f : 1.0f;
    if (e == 0) return s * std::ldexp(static_cast<float>(m), -24);
    return s * std::ldexp(1.0f + m / 1024.0f, e - 15);
}

// llama.cpp qs order -> iq3_xxs_planar_v1 quad-lane order (mirrors tools/artifact/codecs/iq3_xxs.py).
void repack_block(const std::uint8_t* qs, std::uint8_t* out) {
    for (int q = 0; q < 4; ++q) {
        for (int s = 0; s < 8; ++s) {
            out[q * 16 + s * 2]     = qs[s * 8 + 2 * q];
            out[q * 16 + s * 2 + 1] = qs[s * 8 + 2 * q + 1];
        }
    }
    std::uint32_t words[8];
    std::memcpy(words, qs + 64, 32);
    for (int q = 0; q < 4; ++q) {
        std::uint64_t lane = 0;
        for (int s = 0; s < 8; ++s) lane |= static_cast<std::uint64_t>((words[s] >> (7 * q)) & 127u) << (7 * s);
        lane |= static_cast<std::uint64_t>(words[2 * q] >> 28) << 56;
        lane |= static_cast<std::uint64_t>(words[2 * q + 1] >> 28) << 60;
        std::memcpy(out + 64 + q * 8, &lane, 8);
    }
}

struct Iq3Matrix {
    int n, k, blocks;
    std::vector<std::uint8_t> payload; // llama.cpp-order code plane, padding, binary16 d plane
    std::size_t scale_offset;

    Iq3Matrix(int n_, int k_, std::mt19937& rng) : n(n_), k(k_), blocks(k_ / 256) {
        const std::size_t code_bytes = static_cast<std::size_t>(n) * blocks * 96;
        scale_offset                 = (code_bytes + 255) / 256 * 256;
        payload.assign(scale_offset + static_cast<std::size_t>(n) * blocks * 2, 0);
        std::uniform_int_distribution<int> byte(0, 255), mant(0, 1023), exp(4, 6);
        for (std::size_t i = 0; i < code_bytes; ++i) payload[i] = static_cast<std::uint8_t>(byte(rng));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n) * blocks; ++i) {
            const std::uint16_t h = static_cast<std::uint16_t>((exp(rng) << 10) | mant(rng));
            std::memcpy(&payload[scale_offset + i * 2], &h, 2);
        }
    }

    // FP64 decode of one row (reference dequantize_row_iq3_xxs).
    std::vector<double> row(int r) const {
        std::vector<double> y(static_cast<std::size_t>(k));
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* qs = &payload[(static_cast<std::size_t>(r) * blocks + b) * 96];
            std::uint16_t dh;
            std::memcpy(&dh, &payload[scale_offset + (static_cast<std::size_t>(r) * blocks + b) * 2], 2);
            const double d = half_value(dh);
            for (int ib = 0; ib < 8; ++ib) {
                std::uint32_t aux;
                std::memcpy(&aux, qs + 64 + 4 * ib, 4);
                const double db = d * (0.5 + (aux >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    unsigned s7          = (aux >> (7 * l)) & 127u;
                    unsigned parity      = 0;
                    for (unsigned t = s7; t; t >>= 1) parity ^= t & 1u;
                    const unsigned signs = s7 | (parity << 7);
                    const std::uint32_t g1 = kGrid[qs[ib * 8 + 2 * l]];
                    const std::uint32_t g2 = kGrid[qs[ib * 8 + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        const double m1 = (g1 >> (8 * j)) & 255u, m2 = (g2 >> (8 * j)) & 255u;
                        double* out = &y[static_cast<std::size_t>(b) * 256 + ib * 32 + l * 8];
                        out[j]     = db * m1 * ((signs >> j) & 1u ? -1.0 : 1.0);
                        out[j + 4] = db * m2 * ((signs >> (j + 4)) & 1u ? -1.0 : 1.0);
                    }
                }
            }
        }
        return y;
    }

    Weight upload(ninfer::DeviceArena& arena) const {
        std::vector<std::uint8_t> packed(payload);
        for (std::size_t blk = 0; blk < static_cast<std::size_t>(n) * blocks; ++blk) {
            repack_block(&payload[blk * 96], &packed[blk * 96]);
        }
        auto span = arena.alloc_bytes(packed.size(), 256);
        CUDA_CHECK(cudaMemcpy(span.data, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        Weight w;
        w.payload       = span.data;
        w.payload_bytes = payload.size();
        w.qtype         = QType::IQ3_XXS;
        w.layout        = QuantLayout::Iq3XxsPlanar;
        w.group_size    = 256;
        w.group         = 256;
        w.ndim          = 2;
        w.n = w.shape[0] = w.padded_shape[0] = n;
        w.k = w.shape[1] = w.padded_shape[1] = k;
        w.qdata       = span.data;
        w.scales      = static_cast<const std::uint8_t*>(span.data) + scale_offset;
        w.scale_dtype = DType::FP16;
        return w;
    }
};

std::vector<std::uint16_t> random_bf16(std::size_t count, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, sigma);
    std::vector<std::uint16_t> v(count);
    for (auto& b : v) b = bf16_bits(dist(rng));
    return v;
}

Tensor upload_bf16(ninfer::DeviceArena& arena, const std::vector<std::uint16_t>& host, int rows,
                   int cols) {
    Tensor t = arena.alloc(DType::BF16, {rows, cols});
    CUDA_CHECK(cudaMemcpy(t.data, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
    return t;
}

std::vector<std::uint16_t> download_bf16(const Tensor& t, std::size_t count) {
    std::vector<std::uint16_t> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), t.data, count * 2, cudaMemcpyDeviceToHost));
    return host;
}

std::vector<int> sample_rows(int extent) {
    std::vector<int> rows{0, 1, 7, 8, 15, 16, 31};
    for (int i = 1; i < 24; ++i) rows.push_back(extent * i / 24 + (i % 5));
    rows.push_back(extent - 9);
    rows.push_back(extent - 1);
    return rows;
}

int failures = 0;

void check(const char* label, int t, double got, double ref, double scale) {
    const double tol = std::fabs(ref) * (1.0 / 128.0) + scale * 2e-3;
    if (!(std::fabs(got - ref) <= tol)) {
        if (failures < 20) std::printf("  FAIL %s T=%d got %.6f ref %.6f tol %.6f\n", label, t, got, ref, tol);
        ++failures;
    }
}

// Activations as the kernel sees them: BF16 for T <= 32, per-256 absmax int8 for the prefill path
// (mirrors iq3xxs_quantize_act_kernel's FP32 arithmetic).
std::vector<double> kernel_activations(const std::vector<std::uint16_t>& x, int tokens, int k,
                                       bool a8) {
    std::vector<double> v(x.size());
    const bool int8 = a8 && tokens > 32;
    for (std::size_t g = 0; g < x.size(); g += 256) {
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(bf16_value(x[g + i])));
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f, scale = amax / 127.0f;
        for (int i = 0; i < 256; ++i) {
            const float f = bf16_value(x[g + i]);
            v[g + i]      = int8 ? static_cast<double>(std::nearbyint(f * inv)) * scale : f;
        }
    }
    (void)tokens;
    (void)k;
    return v;
}

double dot(const std::vector<double>& w, const std::vector<double>& x, int token, int k) {
    double s = 0.0;
    for (int i = 0; i < k; ++i) s += w[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(token) * k + i];
    return s;
}

double dot(const std::vector<double>& w, const std::vector<std::uint16_t>& x, int token, int k) {
    double s = 0.0;
    for (int i = 0; i < k; ++i) s += w[static_cast<std::size_t>(i)] * bf16_value(x[static_cast<std::size_t>(token) * k + i]);
    return s;
}

double max_quant_err = 0.0; // int8 activation effect vs exact BF16, relative to the output scale

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    std::mt19937 rng(20261010);
    ninfer::DeviceArena arena(std::size_t{512} << 20);
    cudaStream_t stream = nullptr;

    const Iq3Matrix gate_up(34816, 5120, rng);
    const Iq3Matrix down(5120, 17408, rng);
    const Weight w_gate_up = gate_up.upload(arena);
    const Weight w_down    = down.upload(arena);
    const auto swiglu_rows = sample_rows(17408);
    const auto add_rows    = sample_rows(5120);
    std::vector<std::vector<double>> gate_ref, up_ref, down_ref;
    for (int r : swiglu_rows) {
        gate_ref.push_back(gate_up.row(r));
        up_ref.push_back(gate_up.row(r + 17408));
    }
    for (int r : add_rows) down_ref.push_back(down.row(r));

    struct Case {
        int tokens;
        ninfer::ops::LinearPolicy policy;
    };
    std::vector<Case> cases;
    for (int t : {1, 2, 3, 5, 8, 9, 16, 17, 31, 32, 33, 64, 100, 200}) {
        cases.push_back({t, ninfer::ops::LinearPolicy::AllowA8});
    }
    for (int t : {33, 100}) cases.push_back({t, ninfer::ops::LinearPolicy::A16Only});
    for (const Case& c : cases) {
        const int tokens = c.tokens;
        const bool a8    = ninfer::ops::allows_a8(c.policy);
        auto scope       = arena.scope();
        const int before = failures;
        // LinearSwiGLU: out[r] = silu(gate . x) * (up . x)
        const auto x_host = random_bf16(static_cast<std::size_t>(tokens) * 5120, 1.0f, rng);
        const Tensor x    = upload_bf16(arena, x_host, 5120, tokens);
        Tensor out        = arena.alloc(DType::BF16, {17408, tokens});
        ninfer::ops::linear_swiglu(x, w_gate_up, out, c.policy, arena, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got = download_bf16(out, static_cast<std::size_t>(tokens) * 17408);
        const auto xk  = kernel_activations(x_host, tokens, 5120, a8);
        for (std::size_t i = 0; i < swiglu_rows.size(); ++i) {
            for (int t = 0; t < tokens; ++t) {
                const double g   = dot(gate_ref[i], xk, t, 5120);
                const double u   = dot(up_ref[i], xk, t, 5120);
                if (a8 && tokens > 32) {
                    const double ge = dot(gate_ref[i], x_host, t, 5120);
                    const double ue = dot(up_ref[i], x_host, t, 5120);
                    const double e  = std::fabs(ge / (1.0 + std::exp(-ge)) * ue - g / (1.0 + std::exp(-g)) * u);
                    max_quant_err   = std::fmax(max_quant_err, e / (std::fabs(ge) * std::fabs(ue) + 1.0));
                }
                const double ref = g / (1.0 + std::exp(-g)) * u;
                const double scale = std::fabs(g) * std::fabs(u) + std::fabs(u) + 1.0;
                check("swiglu", tokens,
                      bf16_value(got[static_cast<std::size_t>(t) * 17408 + swiglu_rows[i]]), ref, scale);
            }
        }
        // LinearAdd: residual[r] += down . x
        const auto xd_host  = random_bf16(static_cast<std::size_t>(tokens) * 17408, 0.5f, rng);
        const auto res_host = random_bf16(static_cast<std::size_t>(tokens) * 5120, 4.0f, rng);
        const Tensor xd     = upload_bf16(arena, xd_host, 17408, tokens);
        Tensor res          = upload_bf16(arena, res_host, 5120, tokens);
        ninfer::ops::linear_add(xd, w_down, res, c.policy, arena, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got_res = download_bf16(res, static_cast<std::size_t>(tokens) * 5120);
        const auto xdk     = kernel_activations(xd_host, tokens, 17408, a8);
        for (std::size_t i = 0; i < add_rows.size(); ++i) {
            for (int t = 0; t < tokens; ++t) {
                const std::size_t at = static_cast<std::size_t>(t) * 5120 + add_rows[i];
                const double acc     = dot(down_ref[i], xdk, t, 17408);
                const double ref     = acc + bf16_value(res_host[at]);
                check("linear_add", tokens, bf16_value(got_res[at]), ref, std::fabs(acc) + 1.0);
            }
        }
        std::printf("T=%-3d %s\n", tokens, failures == before ? "ok" : "FAILED");
    }
    std::printf("%s (%d mismatches)\n", failures ? "FAILED" : "PASSED", failures);

    // Optional throughput probe: NINFER_IQ3_BENCH=1. Effective bandwidth counts weight bytes only.
    if (const char* bench = std::getenv("NINFER_IQ3_BENCH"); bench && *bench == '1') {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        // Rotate through enough layer copies that every launch streams its weights from DRAM,
        // as the 64-layer decode does; a single copy would stay resident in L2.
        int l2_bytes = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0));
        const std::size_t layer_bytes = gate_up.payload.size() + down.payload.size();
        const int copies = static_cast<int>(4 * static_cast<std::size_t>(l2_bytes) / layer_bytes) + 2;
        ninfer::DeviceArena weights(layer_bytes * copies + (std::size_t{4} << 20));
        std::vector<Weight> gu_copies, dn_copies;
        for (int c = 0; c < copies; ++c) {
            gu_copies.push_back(gate_up.upload(weights));
            dn_copies.push_back(down.upload(weights));
        }
        std::printf("bench: L2 %d MiB, %d layer copies (%.0f MiB)\n", l2_bytes >> 20, copies,
                    layer_bytes * copies / 1048576.0);
        const char* only = std::getenv("NINFER_IQ3_BENCH_TOKENS");
        for (int tokens : {1, 2, 4, 8, 16, 32, 33, 64, 128, 256, 512, 1024, 2048}) {
            if (only && std::atoi(only) != tokens) continue;
            auto scope = arena.scope();
            const Tensor x = upload_bf16(
                arena, random_bf16(static_cast<std::size_t>(tokens) * 5120, 1.0f, rng), 5120, tokens);
            Tensor out      = arena.alloc(DType::BF16, {17408, tokens});
            const Tensor xd = upload_bf16(
                arena, random_bf16(static_cast<std::size_t>(tokens) * 17408, 0.5f, rng), 17408, tokens);
            Tensor res = arena.alloc(DType::BF16, {5120, tokens});
            CUDA_CHECK(cudaMemset(res.data, 0, static_cast<std::size_t>(tokens) * 5120 * 2));
            const auto time_ms = [&](auto&& launch) {
                for (int i = 0; i < 5; ++i) launch();
                const int reps = tokens >= 256 ? 20 : 200;
                CUDA_CHECK(cudaEventRecord(start, stream));
                for (int i = 0; i < reps; ++i) launch();
                CUDA_CHECK(cudaEventRecord(stop, stream));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                return static_cast<double>(ms) / reps;
            };
            int next = 0;
            const double gu = time_ms([&] {
                ninfer::ops::linear_swiglu(x, gu_copies[next++ % copies], out, ninfer::ops::LinearPolicy::AllowA8, arena, stream);
            });
            const double dn = time_ms([&] {
                ninfer::ops::linear_add(xd, dn_copies[next++ % copies], res, ninfer::ops::LinearPolicy::AllowA8, arena, stream);
            });
            const double tflops =
                2.0 * tokens * (34816.0 * 5120 + 5120.0 * 17408) / ((gu + dn) * 1e9);
            std::printf("bench T=%-4d gate_up %7.1f us (%5.0f GB/s)  down %7.1f us (%5.0f GB/s)  "
                        "%5.1f TFLOP/s  64-layer MLP %.2f ms\n",
                        tokens, gu * 1e3, gate_up.payload.size() / (gu * 1e6), dn * 1e3,
                        down.payload.size() / (dn * 1e6), tflops, (gu + dn) * 64);
        }
    }
    return failures ? 1 : 0;
}
