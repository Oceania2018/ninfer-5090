// Q4/Q5 GDN input projection with int8 activations (prefill, LinearPolicy allows A8).

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/rowsplit_q8_mma.cuh"

namespace ninfer::ops::detail {

void q4_q5_gdn_input_q8_launch(const Tensor& x, const Weight& qk_weight,
                               const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& ws, cudaStream_t stream) {
    constexpr std::int32_t kK = 5120, kValueRows = 6144;
    auto scope       = ws.scope();
    const int tokens = x.ne[1];
    const Q8Act act  = q8_quantize_act<kK, Q8ActLayout::Q4Q5Lane>(
        static_cast<const __nv_bfloat16*>(x.data), tokens, ws, stream);
    auto* qkv_out = static_cast<__nv_bfloat16*>(qkv.data);
    const RowSplitGroupedMmaJob jobs[4] = {
        rowsplit_q8_job(qk_weight, 0, qk_weight.n, qkv_out, qkv.ne[0], 0),
        rowsplit_q8_job(value_z_weight, 0, kValueRows, qkv_out, qkv.ne[0], qk_weight.n),
        rowsplit_q8_job(value_z_weight, kValueRows, kValueRows,
                        static_cast<__nv_bfloat16*>(z.data), z.ne[0], 0),
        RowSplitGroupedMmaJob{},
    };
    rowsplit_q8_launch<kK, 3, RowSplitGroupedMmaCodec::Mixed, RowSplitQ8Output::Store>(
        act, jobs, tokens, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
