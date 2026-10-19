#pragma once

#include <cstdint>

namespace ninfer::runtime {

using ContinuationOwnerToken = std::uint64_t;

struct CacheRetentionPriority {
    bool reused = false;
    // Ranking key; larger is hotter. Under LRFU it is the last reference time plus the entry's
    // frequency bonus (ResourceManager::referenced), so it may run ahead of the clock.
    std::uint64_t last_demand = 0;
    // LRFU state at the last reference: its clock and the combined recency-frequency value.
    std::uint64_t referenced_at = 0;
    double crf                  = 0;
};

enum class ReclaimProgress : std::uint8_t { Blocked, Changed, Transferring };

enum class ReclaimPurpose : std::uint8_t { Execution, FreshAdmission, OptionalWrite };

struct ReclaimRights {
    ReclaimPurpose purpose = ReclaimPurpose::Execution;
    std::uint64_t request  = 0;
};

} // namespace ninfer::runtime
