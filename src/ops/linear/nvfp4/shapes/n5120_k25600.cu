#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

#include <stdexcept>

// DFlash2 feature projection (hidden 5120 x five target layers 25600).
// Re-encoded for 16 GB cards without an activation divisor, so the shape is A16-only: no A4
// route and no TMA geometry. Schedules follow the 5120x6144 A16 table.
namespace ninfer::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<5120, 25600>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 8, 1>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 2, 2>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

void reject_a4(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t) {
    throw std::invalid_argument("nvfp4 linear: the 5120x25600 shape has no A4 route");
}

bool never_a4(std::int32_t, std::int32_t) { return false; }

} // namespace

const Nvfp4LinearShape kNvfp4N5120K25600{5120, 25600, launch_a16, reject_a4, never_a4};
} // namespace ninfer::ops::detail
