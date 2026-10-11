// Q4/Q5 attention input projection with int8 activations (prefill, LinearPolicy allows A8).

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/rowsplit_q8_mma.cuh"

namespace ninfer::ops::detail {

void q4_q5_attn_input_q8_launch(const Tensor& x, const Weight& query_key_weight,
                                const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, WorkspaceArena& ws, cudaStream_t stream) {
    constexpr std::int32_t kK = 5120, kQRows = 6144, kKvRows = 1024;
    auto scope       = ws.scope();
    const int tokens = x.ne[1];
    const Q8Act act  = q8_quantize_act<kK, Q8ActLayout::Q4Q5Lane>(
        static_cast<const __nv_bfloat16*>(x.data), tokens, ws, stream);
    const auto out = [](Tensor& t) { return static_cast<__nv_bfloat16*>(t.data); };
    const RowSplitGroupedMmaJob jobs[4] = {
        rowsplit_q8_job(query_key_weight, 0, kQRows, out(q), q.ne[0], 0),
        rowsplit_q8_job(query_key_weight, kQRows, kKvRows, out(k), k.ne[0], 0),
        rowsplit_q8_job(gate_value_weight, 0, kQRows, out(gate), gate.ne[0], 0),
        rowsplit_q8_job(gate_value_weight, kQRows, kKvRows, out(v), v.ne[0], 0),
    };
    rowsplit_q8_launch<kK, 4, RowSplitGroupedMmaCodec::Mixed, RowSplitQ8Output::Store>(
        act, jobs, tokens, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
