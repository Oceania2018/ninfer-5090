// Q5 [5120, 6144] linear_add (attention/GDN output projection) with int8 activations.

#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/rowsplit_q8_mma.cuh"

namespace ninfer::ops::detail {

void q5_linear_add_q8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                             WorkspaceArena& ws, cudaStream_t stream) {
    constexpr std::int32_t kK = 6144;
    auto scope       = ws.scope();
    const int tokens = x.ne[1];
    const Q8Act act  = q8_quantize_act<kK, Q8ActLayout::Q4Q5Lane>(
        static_cast<const __nv_bfloat16*>(x.data), tokens, ws, stream);
    const RowSplitGroupedMmaJob jobs[4] = {
        rowsplit_q8_job(w, 0, w.n, static_cast<__nv_bfloat16*>(residual_out.data),
                        static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16)), 0),
        {}, {}, {},
    };
    rowsplit_q8_launch<kK, 1, RowSplitGroupedMmaCodec::Q5, RowSplitQ8Output::ResidualAdd>(
        act, jobs, tokens, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
