#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Launch-shape tables were tuned on the 170-SM RTX 5090. Cards with far fewer SMs (the 36-SM
// RTX 5060 Ti) keep too few loads in flight with those shapes, so a few small-T dispatches pick
// a second table. One resident GPU per process: the answer is read once and cached.
inline bool small_sm_device() noexcept {
    static const bool small = [] {
        int device = 0;
        int sms    = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            return false;
        }
        return sms <= 48;
    }();
    return small;
}

} // namespace ninfer::ops::detail
