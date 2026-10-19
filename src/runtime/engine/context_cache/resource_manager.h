#pragma once

#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/context_cache/prefix_index.h"
#include "runtime/engine/context_cache/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <unordered_set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Policy owns bounded logical continuations. Native owns checkpoint contents and every byte.
template <class Model>
class ResourceManager {
public:
    using Program    = typename Model::Program;
    using Base       = typename Model::RequestBasePlan;
    using Handle     = typename Model::CheckpointHandle;
    using Source     = typename Model::SourceCandidate;
    using SessionKey = typename Model::CacheSessionKey;
    using OwnerToken = ContinuationOwnerToken;

    using DemandKey =
        typename decltype(std::declval<const Base&>().prefix_shortlist_key(0))::value_type;

    // Optional writes carry only their own demand evidence. Necessary execution has no
    // admission restriction. The request owns base for the lifetime of the reclaim decision.
    struct Admission {
        CacheRetentionPriority priority;
        CacheRetentionPriority demand;
        const Base* base       = nullptr;
        std::uint32_t frontier = 0;
        std::optional<Handle> checkpoint;
        std::vector<Handle> sources;
    };

    struct ReclaimCursor {
        struct Victim {
            Handle handle;
            CacheRetentionPriority priority;
            CacheRetentionPriority demand;
        };

        struct OwnerProtection {
            OwnerToken owner;
            CacheRetentionPriority priority;
            std::uint32_t proven_frontier;
            std::vector<Handle> points;
        };

        // Fixed membership, order and writeback rights for one decision. Native residency
        // supplies progress; a history may still have other pages needed by a later shortage.
        std::vector<Victim> victims;
        std::vector<OwnerProtection> owners;
        std::optional<Admission> admission;
        std::optional<std::uint64_t> gain_limit;
        std::uint64_t sacrificed = 0;
        ReclaimRights rights;
    };

    struct SourceChoice {
        Source source;
        OwnerToken owner           = 0;
        std::uint64_t shared_entry = 0;
        bool take_over             = false;
        std::optional<OwnerToken> resume_owner;
        bool session_hint               = false;
        std::uint64_t publication_order = 0;
        std::uint64_t ordinal           = 0;
    };

    // `ctx_checkpoints` bounds superseded public points per request chain (0 disables it).
    // An agent loop that replaces its last turn every step publishes one public point per step
    // and resumes each step from the previous one exactly once. Without a per-chain bound those
    // adopted-once points stay reuse-tier forever and, once Host is full, starve the next
    // step's capture of admission. See `supersede`.
    //
    // `lrfu_lambda` > 0 ranks retained points by LRFU instead of their last reference alone, and
    // counts a resume as a reference to every shallower public point on its path. See
    // `referenced` and `reference_ancestors`. 0 keeps LRU.
    explicit ResourceManager(bool enabled, ContextMachineCostModel costs,
                             std::uint32_t ctx_checkpoints = 0, double lrfu_lambda = 0)
        : enabled_(enabled), lineage_cap_(ctx_checkpoints), lrfu_lambda_(lrfu_lambda),
          costs_(std::move(costs)) {}

    struct LineageStats {
        std::uint64_t retired_public  = 0;
        std::uint64_t retired_private = 0;
        std::uint64_t boosted_captures = 0;
        // LRFU references credited to shallower public points of a resumed path.
        std::uint64_t ancestor_references = 0;
    };

    [[nodiscard]] const LineageStats& lineage_stats() const { return lineage_stats_; }

    // Disk tier write-behind candidates, in the order reclaim would delete them (so what Host loses
    // first is the most likely to be on disk already), restricted to points a later request can
    // still resume from:
    // - a public point that no deeper point of its chain superseded, and that is either a chain
    //   head or was ever adopted. A superseded point only backs a retry of a step the chain moved
    //   past; an unadopted point outside any chain is a one-off request's own marker.
    // - a private point whose owner was ever adopted, such as a continuing conversation. Other
    //   private points serve only the request that made them.
    // Demand is "ever", not the current reuse tier: balance_reused demotes the oldest demand out
    // of that tier, and those are exactly the points Host is about to lose.
    [[nodiscard]] std::vector<Handle> disk_write_candidates() const {
        std::vector<Handle> out;
        for (const auto handle : victims()) {
            const bool shared = std::any_of(shared_.begin(), shared_.end(), [&](const auto& entry) {
                return entry.handle == handle && !entry.superseded &&
                       (entry.priority.last_demand != 0 || entry.lineage != 0);
            });
            const bool owned = std::any_of(owners_.begin(), owners_.end(), [&](const auto& owner) {
                return owner.priority.last_demand != 0 &&
                       std::any_of(owner.points.begin(), owner.points.end(),
                                   [&](const auto& point) { return point.handle == handle; });
            });
            if (shared || owned) { out.push_back(handle); }
        }
        return out;
    }

    // A request is about to resume from a disk record: that is real demand at this moment.
    [[nodiscard]] Admission restore_admission(const Base& base, std::uint32_t frontier) {
        const CacheRetentionPriority demand{.reused = true, .last_demand = ++clock_};
        return {.priority = demand, .demand = demand, .base = &base, .frontier = frontier};
    }

    // A checkpoint rebuilt from disk enters as a public prefix carrying the restoring demand.
    void publish_restored(Program& program, Handle handle, CacheRetentionPriority priority) {
        if (!enabled_ || !program.valid_checkpoint(handle)) { return; }
        const auto summary = program.checkpoint_metadata(handle);
        shared_.push_back({.handle      = handle,
                           .ordinal     = ++ordinal_,
                           .priority    = priority,
                           .retained_at = ++clock_});
        index_.insert(program, handle, summary.frontier);
        balance_reused(program);
    }

    // Whether resuming from a disk record beats the best resident source. The record costs a
    // disk read, a Host-to-Device copy of the same bytes, and its own remaining prefill.
    [[nodiscard]] bool disk_restore_pays(const Source& best, std::size_t bytes,
                                         std::uint32_t pages, PrefillWork remaining) const {
        const auto saturating = [](std::uint64_t a, std::uint64_t b) {
            return a > UINT64_MAX - b ? UINT64_MAX : a + b;
        };
        const auto resident = saturating(price_context_transfer_requirements(costs_, best.transfers),
                                         costs_.prefill_ns(best.remaining_work));
        const TransferWork copy{.payload_bytes = bytes, .copy_operations = pages + 1U};
        const auto restore =
            saturating(saturating(disk_read_ns(bytes),
                                  costs_.transfer_ns(ContextTransferDirection::HostToDevice, copy)),
                       costs_.prefill_ns(remaining));
        return restore < resident;
    }

    // Generic NVMe read model: 2.5 GB/s plus a fixed submission cost.
    [[nodiscard]] static std::uint64_t disk_read_ns(std::size_t bytes) noexcept {
        constexpr std::uint64_t fixed_ns = 200'000;
        const std::uint64_t streaming    = static_cast<std::uint64_t>(bytes) / 5U * 2U;
        return streaming > UINT64_MAX - fixed_ns ? UINT64_MAX : streaming + fixed_ns;
    }

    [[nodiscard]] std::vector<SourceChoice>
    candidates(Program& program, const Base& base, std::uint32_t maximum_frontier = UINT32_MAX,
               std::optional<OwnerToken> own_resume = std::nullopt, std::uint64_t request = 0,
               std::uint64_t publication_order = UINT64_MAX) {
        prune(program);
        std::vector<SourceChoice> result;
        if (enabled_) {
            for (const auto handle : index_.matches(program, base)) {
                if (program.checkpoint_metadata(handle).frontier > maximum_frontier) { continue; }
                for (const auto& owner : owners_) {
                    const auto point =
                        std::find_if(owner.points.begin(), owner.points.end(),
                                     [&](const auto& p) { return p.handle == handle; });
                    if (point == owner.points.end()) { continue; }
                    const auto& advice = base.context_cache();
                    const bool take =
                        own_resume
                            ? *own_resume == owner.token
                            : !owner.active && owner.publication_order <= publication_order &&
                                  (!owner.session || (advice.update_session_index &&
                                                      advice.session_key == owner.session));
                    result.push_back(
                        {.source            = Source{.checkpoint = handle},
                         .owner             = owner.token,
                         .take_over         = take,
                         .resume_owner      = own_resume,
                         .session_hint      = owner.session && owner.session == advice.session_key,
                         .publication_order = owner.publication_order,
                         .ordinal           = owner.token});
                }
                for (const auto& shared : shared_) {
                    if (shared.handle != handle) { continue; }
                    result.push_back({.source       = Source{.checkpoint = handle},
                                      .shared_entry = shared.ordinal,
                                      .resume_owner = own_resume,
                                      .ordinal      = shared.ordinal});
                }
            }
        }
        result.push_back({.source = Source{}, .resume_owner = own_resume});
        if (const auto* waiting = find_waiting(request);
            waiting && waiting->choice.source.checkpoint &&
            contains(waiting->held, *waiting->choice.source.checkpoint)) {
            result.push_back(waiting->choice);
        }
        std::erase_if(
            result, [&](auto& choice) { return !prepare_source(program, base, choice, request); });
        const auto cost = [&](const Source& source) {
            const auto transfer = price_context_transfer_requirements(costs_, source.transfers);
            const auto prefill  = costs_.prefill_ns(source.remaining_work);
            return transfer > UINT64_MAX - prefill ? UINT64_MAX : transfer + prefill;
        };
        const auto bytes = [](const Source& source) {
            std::uint64_t total = 0;
            for (const auto& transfer : source.transfers) {
                total = transfer.work.payload_bytes > UINT64_MAX - total
                            ? UINT64_MAX
                            : total + transfer.work.payload_bytes;
            }
            return total;
        };
        std::stable_sort(result.begin(), result.end(), [&](const auto& a, const auto& b) {
            const auto left = cost(a.source), right = cost(b.source);
            if (left != right) { return left < right; }
            if (bytes(a.source) != bytes(b.source)) { return bytes(a.source) < bytes(b.source); }
            if (a.source.reused_tokens != b.source.reused_tokens) {
                return a.source.reused_tokens > b.source.reused_tokens;
            }
            if (a.session_hint != b.session_hint) { return a.session_hint; }
            if (bool(a.owner) != bool(b.owner)) { return bool(a.owner); }
            if (a.publication_order != b.publication_order) {
                return a.publication_order > b.publication_order;
            }
            return a.ordinal < b.ordinal;
        });
        // Equal handles are the same Native contents. Independently computed states with equal
        // semantic keys remain separate candidates; Native must not splice their KV and state.
        std::vector<SourceChoice> unique;
        for (auto& choice : result) {
            const auto duplicate =
                std::any_of(unique.begin(), unique.end(), [&](const auto& prior) {
                    return prior.source.checkpoint == choice.source.checkpoint &&
                           prior.source.consume_source == choice.source.consume_source &&
                           prior.source.private_points == choice.source.private_points;
                });
            if (!duplicate) { unique.push_back(std::move(choice)); }
        }
        return unique;
    }

    // Evaluation grants consumption rights but does not retire anything. Native checks the
    // complete binding before applying these grants in its ownership handoff.
    [[nodiscard]] bool prepare_source(Program& program, const Base& base, SourceChoice& choice,
                                      std::uint64_t request                   = 0,
                                      std::optional<ReclaimRights> revocation = std::nullopt) {
        const auto selected = choice.source.checkpoint;
        if (selected && !program.valid_checkpoint(*selected)) { return false; }
        auto* owner = find_owner(choice.owner);
        if (choice.take_over && (!owner || owner->publication_order != choice.publication_order ||
                                 (owner->active && choice.resume_owner != choice.owner))) {
            choice.take_over = false;
        }
        std::vector<Handle> retired;
        if (choice.take_over && selected && owner) {
            const auto frontier = program.checkpoint_metadata(*selected).frontier;
            for (const auto& point : owner->points) {
                if (point.handle == *selected || !program.valid_checkpoint(point.handle)) {
                    continue;
                }
                const auto summary = program.checkpoint_metadata(point.handle);
                if (summary.frontier > frontier && !summary.leased &&
                    consumable(point.handle, owner->token, request, revocation)) {
                    retired.push_back(point.handle);
                }
            }
        }
        auto carried = points(choice.resume_owner.value_or(choice.owner));
        if (const auto* waiting = find_waiting(request);
            waiting && waiting->choice.source.checkpoint == selected) {
            carried = waiting->choice.source.private_points;
            std::erase_if(carried, [&](auto point) { return !contains(waiting->held, point); });
        }
        const auto private_owner =
            choice.resume_owner.value_or(choice.take_over ? choice.owner : 0);
        const bool consume = selected && consumable(*selected, private_owner, request, revocation);
        auto inspected     = program.inspect_source(base, selected, consume, carried, retired);
        if (!inspected) { return false; }
        inspected->take_private =
            std::all_of(inspected->private_points.begin(), inspected->private_points.end(),
                        [&](auto point) { return consumable(point, private_owner, request); });
        choice.source = std::move(*inspected);
        return true;
    }

    void retain_source(Program& program, std::uint64_t request, const SourceChoice& choice) {
        std::vector<Handle> held;
        if (choice.source.checkpoint) { held.push_back(*choice.source.checkpoint); }
        for (const auto point : choice.source.private_points) {
            if (!contains(held, point)) { held.push_back(point); }
        }
        auto* waiting = find_waiting(request);
        if (!waiting) {
            if (held.empty()) { return; }
            waiting_.push_back({.request = request, .choice = choice, .held = std::move(held)});
            return;
        }
        auto old        = std::move(waiting->held);
        waiting->choice = choice;
        waiting->held   = std::move(held);
        for (const auto point : old) { release_unreferenced(program, point); }
    }

    [[nodiscard]] std::optional<SourceChoice> retained_source(std::uint64_t request) const {
        const auto* waiting = find_waiting(request);
        if (!waiting || !waiting->choice.source.checkpoint ||
            !contains(waiting->held, *waiting->choice.source.checkpoint)) {
            return std::nullopt;
        }
        return waiting->choice;
    }

    [[nodiscard]] bool has_source_record(std::uint64_t request) const {
        return find_waiting(request) != nullptr;
    }

    [[nodiscard]] bool can_transfer_private(OwnerToken owner, std::uint64_t request) const {
        const auto carried = points(owner);
        return std::all_of(carried.begin(), carried.end(),
                           [&](auto point) { return consumable(point, owner, request); });
    }

    [[nodiscard]] std::uint32_t source_revocations(std::uint64_t request) const {
        const auto* waiting = find_waiting(request);
        return waiting ? waiting->revocations : 0;
    }

    void binding_started(std::uint64_t request, std::span<const Handle> retired,
                         std::optional<Handle> consumed) {
        if (auto* waiting = find_waiting(request)) { waiting->binding = true; }
        for (const auto handle : retired) { forget(handle); }
        if (consumed) {
            for (auto& waiting : waiting_) {
                if (waiting.request != request && std::erase(waiting.held, *consumed)) {
                    ++waiting.revocations;
                }
            }
        }
    }

    void release_source(Program& program, std::uint64_t request) {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(), [=](const auto& entry) {
            return entry.request == request;
        });
        if (found == waiting_.end()) { return; }
        auto held = std::move(found->held);
        waiting_.erase(found);
        for (const auto handle : held) { release_unreferenced(program, handle); }
    }

    // Called exactly once after Native's irreversible bind commit. The choice preserves logical
    // ownership even when Native consumed its original checkpoint handle during that commit.
    [[nodiscard]] OwnerToken adopt(Program& program, const SourceChoice& choice, const Base& base,
                                   std::uint64_t publication_order,
                                   std::span<const Handle> carried = {},
                                   std::span<const Handle> retired = {}) {
        for (const auto handle : retired) { forget(handle); }
        if (!enabled_ || !base.summary().publish_continuation) {
            for (const auto handle : carried) { release_unreferenced(program, handle); }
            return 0;
        }
        OwnerToken token = 0;
        CacheRetentionPriority priority;
        std::uint32_t proven_frontier = 0;
        Shared* adopted_shared        = nullptr;
        if (choice.resume_owner) {
            token = *choice.resume_owner;
            if (!find_owner(token)) { throw std::logic_error("resume lost continuation owner"); }
        } else {
            if (choice.source.reused_tokens) {
                const auto used_at = ++clock_;
                if (const auto key = base.prefix_shortlist_key(choice.source.reused_tokens)) {
                    remember_demand(*key, 0,
                                    CacheRetentionPriority{.reused = true, .last_demand = used_at});
                }
                reference_ancestors(program, base, choice.source.reused_tokens, used_at,
                                    choice.shared_entry);
                if (auto* owner = find_owner(choice.owner)) {
                    // The donor's history is referenced again; a fork carries the same evidence.
                    priority        = referenced(owner->priority, used_at);
                    proven_frontier = choice.source.reused_tokens;
                    // Consuming a branch replaces its proof. A shallower independent
                    // Fork has its own new evidence and leaves its donor's pair intact.
                    if (choice.take_over || proven_frontier >= owner->proven_frontier) {
                        owner->priority        = priority;
                        owner->proven_frontier = proven_frontier;
                    }
                    if (choice.take_over) { token = owner->token; }
                } else if (auto* shared = find_shared(choice.shared_entry)) {
                    shared->priority = referenced(shared->priority, used_at);
                    adopted_shared   = shared;
                }
            }
            if (!token) {
                token = ++ordinal_;
                owners_.push_back({.token           = token,
                                   .priority        = priority,
                                   .proven_frontier = proven_frontier,
                                   .retained_at     = ++clock_});
            }
        }
        auto* owner              = find_owner(token);
        owner->active            = true;
        owner->publication_order = publication_order;
        if (lineage_cap_ && !choice.resume_owner) {
            note_lineage(program, *owner, adopted_shared, choice.source.reused_tokens);
        }
        replace_points(program, *owner, carried);
        prune(program);
        balance_reused(program);
        return token;
    }

    void publish(Program& program, OwnerToken token, Handle handle) {
        if (!program.valid_checkpoint(handle)) { return; }
        const auto summary = program.checkpoint_metadata(handle);
        if (!enabled_ || !token) {
            release_unreferenced(program, handle);
            return;
        }
        if (summary.role == CheckpointRole::SharedPrefix) {
            if (std::none_of(shared_.begin(), shared_.end(),
                             [&](const auto& entry) { return entry.handle == handle; })) {
                shared_.push_back(
                    {.handle   = handle,
                     .ordinal  = ++ordinal_,
                     .priority = demand_priority(program.checkpoint_key(handle, summary.frontier)),
                     .retained_at = ++clock_});
                index_.insert(program, handle, summary.frontier);
                if (lineage_cap_) {
                    if (const auto* owner = find_owner(token);
                        owner && owner->lineage && summary.frontier > owner->adopted_frontier) {
                        auto& entry     = shared_.back();
                        entry.lineage   = owner->lineage;
                        entry.parent    = owner->adopted_shared;
                        entry.publisher = token;
                        // The chain's head is what the next step resumes from. Fresh and not yet
                        // adopted it would be the cheapest victim in Host (its loss is only one
                        // step over the point behind it), so one-off requests between steps
                        // evicted it every time. It carries the demand of the point it extends.
                        if (const auto* adopted = find_shared(owner->adopted_shared);
                            adopted && adopted->priority.reused && !entry.priority.reused) {
                            entry.priority = adopted->priority;
                        }
                        // `entry` may move below; it is not used again.
                        retire_dead_siblings(program, owner->lineage, token,
                                             owner->adopted_frontier);
                        supersede(program, owner->adopted_shared, owner->lineage);
                    }
                }
            }
        } else {
            auto* owner = find_owner(token);
            if (!owner) {
                throw std::logic_error("checkpoint publication lost continuation owner");
            }
            const auto observed = demand_priority(program.checkpoint_key(handle, summary.frontier));
            if (summary.role == CheckpointRole::InputReplay && observed.reused &&
                (!owner->priority.reused || observed.last_demand > owner->priority.last_demand)) {
                owner->priority        = observed;
                owner->proven_frontier = summary.frontier;
            }
            if (std::none_of(owner->points.begin(), owner->points.end(),
                             [&](const auto& p) { return p.handle == handle; })) {
                std::optional<Handle> replace;
                std::size_t anchors = 0;
                std::optional<Handle> oldest_anchor;
                for (const auto& point : owner->points) {
                    if (point.role != summary.role) { continue; }
                    if (summary.role != CheckpointRole::LongAnchor) {
                        replace = point.handle;
                        break;
                    }
                    ++anchors;
                    if (!oldest_anchor) { oldest_anchor = point.handle; }
                    if (program.checkpoint_metadata(point.handle).frontier == summary.frontier) {
                        replace = point.handle;
                    }
                }
                if (!replace && anchors >= 4) { replace = oldest_anchor; }
                if (replace && !remove_point(program, *owner, *replace)) {
                    // Optional replacement never turns a legitimate request into a failure.
                    release_unreferenced(program, handle);
                    return;
                }
                owner->points.push_back({handle, summary.role});
                index_.insert(program, handle, summary.frontier);
            }
        }
        balance_reused(program);
    }

    void finish(Program& program, OwnerToken token, const Base& base,
                std::uint64_t publication_order) {
        auto* owner = find_owner(token);
        if (!owner) { return; }
        owner->active            = false;
        owner->publication_order = publication_order;
        const auto& advice       = base.context_cache();
        if (advice.update_session_index && advice.session_key) {
            const auto newer = std::any_of(owners_.begin(), owners_.end(), [&](const auto& other) {
                return other.token != token && other.session == advice.session_key &&
                       other.publication_order > publication_order;
            });
            if (!newer) {
                for (auto& other : owners_) {
                    if (other.session == advice.session_key) { other.session.reset(); }
                }
                owner->session = advice.session_key;
            }
        }
        prune(program);
        balance_reused(program);
    }

    void abandon(Program& program, OwnerToken token) {
        auto* owner = find_owner(token);
        if (!owner) { return; }
        const auto old = points(token);
        std::erase_if(owners_, [&](const auto& entry) { return entry.token == token; });
        for (const auto handle : old) { release_unreferenced(program, handle); }
    }

    [[nodiscard]] std::vector<Handle> points(OwnerToken token) const {
        std::vector<Handle> out;
        if (const auto* owner = find_owner(token)) {
            for (const auto& point : owner->points) { out.push_back(point.handle); }
        }
        return out;
    }

    [[nodiscard]] CacheRetentionPriority retention(OwnerToken token) const {
        const auto* owner = find_owner(token);
        return owner ? owner->priority : CacheRetentionPriority{};
    }

    bool recycle_input(Program& program, OwnerToken token) {
        auto* owner = find_owner(token);
        if (!owner) { return false; }
        const auto point =
            std::find_if(owner->points.begin(), owner->points.end(),
                         [](const auto& p) { return p.role == CheckpointRole::InputReplay; });
        if (point == owner->points.end()) { return false; }
        const auto handle = point->handle;
        if (program.valid_checkpoint(handle) && program.checkpoint_metadata(handle).leased) {
            return false;
        }
        return remove_point(program, *owner, handle);
    }

    // The caller supplies each request's strictly advancing, non-overlapping committed
    // prefill ranges. Replay and preparation never call this; one candidate is crossed at
    // most once by a request, including when other requests commit between its chunks.
    void observe_committed(const Base& base, std::uint64_t request_id, std::uint32_t begin,
                           std::uint32_t end) {
        if (!enabled_ || begin >= end) { return; }
        for (const auto frontier : base.capture_frontiers()) {
            if (frontier <= begin || frontier > end) { continue; }
            const auto key = base.prefix_shortlist_key(frontier);
            if (!key) { continue; }
            remember_demand(*key, request_id);
        }
    }

    // request_id names the capturing request. A capture is decided before its own request's
    // commit is observed, so when another request already crossed the same prefix, this
    // crossing is the second one: an agent task's next step capturing what its previous step
    // left behind. Without it, that step's points carry no demand under a full Host, the step
    // after misses, and the task never earns the demand it would have shown.
    [[nodiscard]] Admission capture_admission(Program&, OwnerToken owner, const Base& base,
                                              std::uint32_t frontier,
                                              bool inherit_private = true,
                                              bool input_capture   = false,
                                              std::optional<std::uint64_t> request_id = {}) const {
        const auto key = base.prefix_shortlist_key(frontier);
        auto demand    = demand_priority(key);
        if (!demand.reused && key && request_id) {
            const auto seen = std::find_if(demand_.begin(), demand_.end(), [&](const auto& entry) {
                return entry.key == *key && entry.request_id != 0 &&
                       entry.request_id != *request_id;
            });
            if (seen != demand_.end()) {
                demand = {.reused = true, .last_demand = seen->observed_at};
            }
        }
        // Extending the non-root public point this request resumed from is the chain's next
        // step: in an agent loop it is what the next request resumes from. It carries the
        // adopted point's demand instead of the one-step coverage budget, which a full Host of
        // older history otherwise never lets it meet. The input point at the end of the prompt
        // is excluded: that is the turn the next step replaces.
        if (lineage_cap_ && !demand.reused && !input_capture) {
            if (const auto* entry = find_owner(owner);
                entry && entry->adopted_shared && frontier > entry->adopted_frontier) {
                if (const auto* adopted = find_shared(entry->adopted_shared);
                    adopted && adopted->parent && adopted->priority.reused) {
                    demand = adopted->priority;
                    ++lineage_stats_.boosted_captures;
                }
            }
        }
        // A recurrent continuation's heat can rank its owner, but a new optional point
        // has not itself demonstrated repeat demand. Its added coverage remains bounded.
        auto priority = demand;
        if (!demand.reused && inherit_private) { priority = retention(owner); }
        return {.priority = priority, .demand = demand, .base = &base, .frontier = frontier};
    }

    [[nodiscard]] ReclaimCursor begin_reclaim(Program& program,
                                              std::optional<Admission> admission = std::nullopt,
                                              ReclaimRights rights               = {}) {
        prune(program);
        balance_reused(program);
        auto cursor      = reclaim_order(program);
        cursor.admission = admission;
        cursor.rights    = admission ? ReclaimRights{ReclaimPurpose::OptionalWrite} : rights;
        if (admission && !admission->demand.reused) {
            cursor.gain_limit = recovery_gain(program, *admission, surviving(program, {}));
        }
        return cursor;
    }

    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims(Program& program, std::size_t bytes, std::optional<Handle> incoming,
                 std::span<const Handle> later_snapshots = {},
                 std::span<const Handle> excluded        = {},
                 std::optional<Admission> admission      = std::nullopt,
                 const ReclaimCursor* decision = nullptr, ReclaimRights rights = {}) {
        prune(program);
        Evaluation evaluation;
        if (decision) {
            return host_victims_in_order(program, bytes, incoming, later_snapshots, excluded,
                                         decision->admission, *decision, evaluation);
        }
        auto cursor   = reclaim_order(program);
        cursor.rights = admission ? ReclaimRights{ReclaimPurpose::OptionalWrite} : rights;
        return host_victims_in_order(program, bytes, incoming, later_snapshots, excluded, admission,
                                     cursor, evaluation);
    }

    void commit_host_victims(Program& program, std::span<const Handle> handles,
                             ReclaimCursor& cursor) {
        commit_release(program, handles, cursor);
    }

    bool erase(Program& program, Handle handle, ReclaimRights rights = {}) {
        if (!references(handle)) { return false; }
        if (!may_revoke({&handle, 1}, rights)) { return false; }
        if (program.valid_checkpoint(handle) && !program.release_checkpoint(handle)) {
            return false;
        }
        forget(handle);
        prune(program);
        return true;
    }

    [[nodiscard]] ReclaimProgress reclaim(Program& program, ContextResourceUsage shortage,
                                          std::span<const Handle> excluded = {}) {
        if (program.has_context_transaction()) { return ReclaimProgress::Transferring; }
        auto cursor = begin_reclaim(program);
        return reclaim(program, shortage, excluded, cursor);
    }

    [[nodiscard]] ReclaimProgress reclaim(Program& program, ContextResourceUsage shortage,
                                          std::span<const Handle> excluded, ReclaimCursor& cursor) {
        if (program.has_context_transaction()) { return ReclaimProgress::Transferring; }
        prune(program);
        if (shortage.host_bytes && program.release_redundant_host(excluded)) {
            return ReclaimProgress::Changed;
        }
        std::vector<Handle> allowed;
        for (const auto& victim : cursor.victims) {
            if (references(victim.handle) && program.valid_checkpoint(victim.handle) &&
                !contains(excluded, victim.handle)) {
                allowed.push_back(victim.handle);
            }
        }
        if (shortage.host_bytes) {
            Evaluation evaluation;
            const auto victims =
                host_victims_in_order(program, shortage.host_bytes, std::nullopt, {}, excluded,
                                      cursor.admission, cursor, evaluation, true);
            if (!victims) { return ReclaimProgress::Blocked; }
            commit_release(program, *victims, cursor);
            return victims->empty() ? ReclaimProgress::Blocked : ReclaimProgress::Changed;
        }
        auto plans = program.plan_reclaim(allowed, excluded, shortage);
        Evaluation evaluation{.native = &plans};
        using Demotion = typename decltype(plans.demotions)::value_type;
        using Release  = typename decltype(plans.releases)::value_type;

        struct Action {
            std::optional<Demotion> demotion;
            std::optional<Release> release;
            std::vector<Handle> host_victims;
            std::optional<Admission> preservation;
            ActionRank rank;
        };

        const auto usage     = program.physical_usage();
        const auto free_host = usage.capacity.host_bytes - usage.occupied.host_bytes;
        const bool needs_host_cleanup =
            std::any_of(plans.demotions.begin(), plans.demotions.end(),
                        [&](const auto& plan) { return plan.host_bytes > free_host; });
        const bool existing_host_destination =
            std::any_of(plans.demotions.begin(), plans.demotions.end(),
                        [](const auto& plan) { return plan.host_bytes == 0; });
        // Redundant-replica cleanup is a distinct capacity change. Quotes below are read
        // only and never observe side effects from evaluating another candidate.
        if (needs_host_cleanup && !existing_host_destination &&
            program.release_redundant_host(excluded)) {
            return ReclaimProgress::Changed;
        }

        std::vector<Action> actions;
        actions.reserve(plans.demotions.size() + plans.releases.size());
        for (auto& plan : plans.demotions) {
            const auto units = useful(plan.released, shortage);
            if (!units) { continue; }
            auto facts = action_facts(program, plan.sources, cursor, evaluation);
            if (!admits(program, plan.sources, facts, cursor.admission, cursor, evaluation, true)) {
                continue;
            }
            auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
            protected_sources.insert(protected_sources.end(), plan.sources.begin(),
                                     plan.sources.end());
            auto preservation =
                preservation_admission(program, plan.sources, cursor, facts.priority);
            auto victims =
                host_victims_in_order(program, plan.host_bytes, preservation.checkpoint, {},
                                      protected_sources, preservation, cursor, evaluation);
            if (!victims) { continue; }
            auto rank       = rank_action(program, plan.sources, facts, units, evaluation, true);
            auto host_facts = action_facts(program, *victims, cursor, evaluation);
            const auto host_rank = rank_action(program, *victims, host_facts, units, evaluation);
            rank.loss            = host_rank.loss;
            rank.waiting         = host_rank.waiting;
            rank.order           = std::max(rank.order, host_rank.order);
            if (host_rank.priority.reused &&
                (!rank.priority.reused ||
                 host_rank.priority.last_demand > rank.priority.last_demand)) {
                rank.priority = host_rank.priority;
            }
            actions.push_back({.demotion     = std::move(plan),
                               .host_victims = std::move(*victims),
                               .preservation = std::move(preservation),
                               .rank         = rank});
        }
        for (auto& plan : plans.releases) {
            const auto units = useful(plan.released, shortage);
            if (!units) { continue; }
            auto facts = action_facts(program, plan.sources, cursor, evaluation);
            if (!admits(program, plan.sources, facts, cursor.admission, cursor, evaluation)) {
                continue;
            }
            auto rank = rank_action(program, plan.sources, facts, units, evaluation);
            actions.push_back({.release = std::move(plan), .rank = rank});
        }
        std::stable_sort(actions.begin(), actions.end(),
                         [](const auto& a, const auto& b) { return less_rank(a.rank, b.rank); });
        for (std::size_t index = 0; index < actions.size(); ++index) {
            const auto& action = actions[index];
            if (action.demotion) {
                if (!shortage.state_slots && action.host_victims.empty()) {
                    // Preserve the ranked action order. Independent KV groups can share
                    // one transfer transaction while every holder keeps its own rights.
                    auto builder           = plans.begin_kv_batch(shortage);
                    std::size_t components = 0;
                    std::uint64_t released = 0;
                    const auto needed =
                        shortage.main_kv_pages ? shortage.main_kv_pages : shortage.backend_kv_pages;
                    for (std::size_t next = index; next < actions.size() && released < needed;
                         ++next) {
                        const auto& candidate = actions[next];
                        if (!candidate.demotion) {
                            // These are alternative ways to reclaim physical resources.
                            // A delete covered by earlier migrations adds no pages to this
                            // deficit; keep it in actions for failure fallback, not the batch.
                            if (builder.covers(*candidate.release)) { continue; }
                            break;
                        }
                        if (!candidate.host_victims.empty()) { break; }
                        if (!builder.append(*candidate.demotion)) { break; }
                        ++components;
                        released += useful(candidate.demotion->released, shortage);
                    }
                    if (components > 1) {
                        if (const auto batch = builder.finish()) {
                            auto facts = action_facts(program, batch->sources, cursor, evaluation);
                            if (admits(program, batch->sources, facts, cursor.admission, cursor,
                                       evaluation, true) &&
                                program.start_demote(*batch)) {
                                return program.has_context_transaction()
                                           ? ReclaimProgress::Transferring
                                           : ReclaimProgress::Changed;
                            }
                        }
                        // A combined reservation may fail without changing any replica.
                        // The original first action remains the next legal choice.
                    }
                }
                const auto& plan       = *action.demotion;
                auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
                protected_sources.insert(protected_sources.end(), plan.sources.begin(),
                                         plan.sources.end());
                const auto current = host_victims_in_order(
                    program, plan.host_bytes, action.preservation->checkpoint, {},
                    protected_sources, action.preservation, cursor, evaluation);
                if (!current || *current != action.host_victims) { continue; }
                commit_release(program, action.host_victims, cursor, action.rank.loss);
                if (program.start_demote(plan)) {
                    return program.has_context_transaction() ? ReclaimProgress::Transferring
                                                             : ReclaimProgress::Changed;
                }
                if (!action.host_victims.empty()) { return ReclaimProgress::Changed; }
                continue;
            }
            // Native quoted every holder required for this actual physical release.
            commit_release(program, action.release->sources, cursor, action.rank.loss);
            return ReclaimProgress::Changed;
        }
        return ReclaimProgress::Blocked;
    }

    void release_all(Program& program) noexcept {
        const auto handles = victims();
        index_.clear();
        owners_.clear();
        shared_.clear();
        waiting_.clear();
        demand_.clear();
        for (const auto handle : handles) {
            if (program.valid_checkpoint(handle)) { (void)program.release_checkpoint(handle); }
        }
    }

private:
    struct WaitingSource {
        std::uint64_t request;
        SourceChoice choice;
        std::vector<Handle> held;
        std::uint32_t revocations = 0;
        bool binding              = false;
    };

    WaitingSource* find_waiting(std::uint64_t request) {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(),
                                        [=](const auto& item) { return item.request == request; });
        return found == waiting_.end() ? nullptr : &*found;
    }

    const WaitingSource* find_waiting(std::uint64_t request) const {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(),
                                        [=](const auto& item) { return item.request == request; });
        return found == waiting_.end() ? nullptr : &*found;
    }

    bool consumable(Handle handle, OwnerToken owner, std::uint64_t request,
                    std::optional<ReclaimRights> revocation = std::nullopt) const {
        std::size_t allowed = 0;
        if (const auto* entry = find_owner(owner)) {
            allowed = std::count_if(entry->points.begin(), entry->points.end(),
                                    [&](const auto& point) { return point.handle == handle; });
        }
        if (catalog_reference_count(handle) != allowed) { return false; }
        for (const auto& waiting : waiting_) {
            if (!contains(waiting.held, handle)) { continue; }
            if (waiting.binding) { return false; }
            if (waiting.request != request &&
                (!revocation || revocation->purpose == ReclaimPurpose::OptionalWrite ||
                 (revocation->purpose == ReclaimPurpose::FreshAdmission &&
                  waiting.request <= revocation->request))) {
                return false;
            }
            ++allowed;
        }
        return allowed != 0;
    }

    [[nodiscard]] std::uint64_t waiting_ticket(std::span<const Handle> handles) const {
        std::uint64_t oldest = 0;
        for (const auto& waiting : waiting_) {
            if (std::any_of(handles.begin(), handles.end(),
                            [&](auto handle) { return contains(waiting.held, handle); })) {
                oldest = oldest ? std::min(oldest, waiting.request) : waiting.request;
            }
        }
        return oldest;
    }

    bool may_revoke(std::span<const Handle> handles, ReclaimRights rights) const {
        for (const auto& waiting : waiting_) {
            if (!std::any_of(handles.begin(), handles.end(),
                             [&](auto handle) { return contains(waiting.held, handle); })) {
                continue;
            }
            if (waiting.binding || rights.purpose == ReclaimPurpose::OptionalWrite ||
                (rights.purpose == ReclaimPurpose::FreshAdmission &&
                 waiting.request <= rights.request)) {
                return false;
            }
        }
        return true;
    }

    struct Demand {
        DemandKey key;
        std::uint64_t request_id;
        CacheRetentionPriority priority;
        std::uint64_t observed_at;
    };

    void remember_demand(const DemandKey& key, std::uint64_t request_id,
                         std::optional<CacheRetentionPriority> adopted = std::nullopt) {
        auto found = std::find_if(demand_.begin(), demand_.end(),
                                  [&](const auto& entry) { return entry.key == key; });
        if (found != demand_.end()) {
            if (!adopted && found->request_id == request_id) { return; }
            found->request_id = request_id;
            found->priority =
                adopted.value_or(CacheRetentionPriority{.reused = true, .last_demand = clock_ + 1});
            if (!adopted) { ++clock_; }
            found->observed_at = clock_;
            return;
        }
        if (demand_.size() == kDemandCapacity) {
            const auto oldest =
                std::min_element(demand_.begin(), demand_.end(), [](const auto& a, const auto& b) {
                    return a.observed_at < b.observed_at;
                });
            demand_.erase(oldest);
        }
        demand_.push_back({key, request_id, adopted.value_or(CacheRetentionPriority{}), ++clock_});
    }

    [[nodiscard]] CacheRetentionPriority demand_priority(std::optional<DemandKey> key) const {
        if (!key) { return {}; }
        const auto found = std::find_if(demand_.begin(), demand_.end(),
                                        [&](const auto& entry) { return entry.key == *key; });
        return found == demand_.end() ? CacheRetentionPriority{} : found->priority;
    }

    struct Point {
        Handle handle;
        CheckpointRole role;
    };

    struct Owner {
        OwnerToken token;
        CacheRetentionPriority priority;
        std::uint32_t proven_frontier   = 0;
        std::uint64_t retained_at       = 0;
        std::uint64_t publication_order = 0;
        bool active                     = false;
        std::optional<SessionKey> session;
        std::vector<Point> points;
        // The chain this request's public captures extend, and the public point it resumed
        // from (0 when it started from a private point or from the root).
        std::uint64_t lineage          = 0;
        std::uint64_t adopted_shared   = 0;
        std::uint32_t adopted_frontier = 0;
    };

    struct Shared {
        Handle handle;
        std::uint64_t ordinal;
        CacheRetentionPriority priority;
        std::uint64_t retained_at;
        std::uint64_t lineage = 0;
        // The public point the publishing request resumed from. Zero marks a root: a point
        // published without resuming from public history (system prompt, run goal) is never
        // superseded by the chains that grow from it.
        std::uint64_t parent      = 0;
        OwnerToken publisher      = 0;
        std::uint32_t adoptions   = 0;
        std::uint64_t superseded  = 0;  // clock when a deeper point of its chain replaced it
    };

    static bool contains(std::span<const Handle> handles, Handle handle) {
        return std::find(handles.begin(), handles.end(), handle) != handles.end();
    }

    // Membership for action sources tested against every victim, owner point and survivor.
    // Hashing the slot index keeps those per-candidate scans linear in the cache size;
    // equality stays the handle's own comparison.
    struct HandleIndexHash {
        std::size_t operator()(const Handle& handle) const noexcept {
            return std::hash<std::uint32_t>{}(handle.index);
        }
    };
    using HandleSet = std::unordered_set<Handle, HandleIndexHash>;

    static HandleSet handle_set(std::span<const Handle> handles) {
        return HandleSet(handles.begin(), handles.end());
    }

    Owner* find_owner(OwnerToken token) {
        const auto it = std::find_if(owners_.begin(), owners_.end(),
                                     [&](const auto& owner) { return owner.token == token; });
        return it == owners_.end() ? nullptr : &*it;
    }

    const Owner* find_owner(OwnerToken token) const {
        const auto it = std::find_if(owners_.begin(), owners_.end(),
                                     [&](const auto& owner) { return owner.token == token; });
        return it == owners_.end() ? nullptr : &*it;
    }

    Shared* find_shared(std::uint64_t ordinal) {
        const auto it = std::find_if(shared_.begin(), shared_.end(),
                                     [&](const auto& entry) { return entry.ordinal == ordinal; });
        return it == shared_.end() ? nullptr : &*it;
    }

    const Shared* find_shared(std::uint64_t ordinal) const {
        const auto it = std::find_if(shared_.begin(), shared_.end(),
                                     [&](const auto& entry) { return entry.ordinal == ordinal; });
        return it == shared_.end() ? nullptr : &*it;
    }

    // LRFU (Lee et al., 2001). An entry's combined recency-frequency value is
    // CRF(t) = sum over its references t_i of 2^(-lambda (t - t_i)). Between references every
    // entry decays by the same factor, so ranking by CRF at a common time ranks by the constant
    // t_last + log2(CRF(t_last)) / lambda. That key is stored in last_demand, so every existing
    // comparison of demand ranks by LRFU unchanged; a single reference gives exactly t_last.
    // The time unit is a cache event (clock_), about ten per CUA step with its side requests.
    // The largest bonus, for an entry referenced on every event, is about
    // log2(1 / (lambda ln 2)) / lambda events.
    [[nodiscard]] CacheRetentionPriority referenced(CacheRetentionPriority prior,
                                                    std::uint64_t now) const {
        CacheRetentionPriority next{
            .reused = true, .last_demand = now, .referenced_at = now, .crf = 1};
        if (lrfu_lambda_ <= 0) { return next; }
        if (prior.crf > 0 && now >= prior.referenced_at) {
            next.crf += prior.crf * std::exp2(-lrfu_lambda_ *
                                              static_cast<double>(now - prior.referenced_at));
        }
        next.last_demand = now + static_cast<std::uint64_t>(std::log2(next.crf) / lrfu_lambda_);
        return next;
    }

    // A resume of `reused` tokens also used every shallower public point on its path (the system
    // prompt, the task). Under LRU those age out while each run resumes from its own deeper
    // history, and the next run then prefills its whole prefix cold. Superseded chain points are
    // left demoted: their contents are covered by the head that replaced them.
    void reference_ancestors(Program& program, const Base& base, std::uint32_t reused,
                             std::uint64_t now, std::uint64_t adopted_shared) {
        if (lrfu_lambda_ <= 0) { return; }
        for (const auto handle : index_.matches(program, base)) {
            if (!program.valid_checkpoint(handle) ||
                program.checkpoint_metadata(handle).frontier >= reused) {
                continue;
            }
            for (auto& entry : shared_) {
                if (entry.handle == handle && entry.ordinal != adopted_shared && !entry.superseded) {
                    entry.priority = referenced(entry.priority, now);
                    ++lineage_stats_.ancestor_references;
                }
            }
        }
    }

    // Where this request sits in a chain of public points. Resuming from a non-root public
    // point continues that point's chain; resuming from a root (or from nothing public) starts
    // a new one, so concurrent agent runs sharing a system prompt keep separate chains.
    void note_lineage(Program& program, Owner& owner, Shared* adopted, std::uint32_t reused) {
        owner.adopted_shared   = 0;
        owner.adopted_frontier = reused;
        if (!adopted) {
            owner.lineage = ++ordinal_;
            return;
        }
        // A second reader makes the point a branch: it is no longer one step's leftover.
        if (++adopted->adoptions > 1) { adopted->superseded = 0; }
        owner.adopted_shared = adopted->ordinal;
        owner.lineage        = adopted->parent ? adopted->lineage : ++ordinal_;
        if (!adopted->parent || !adopted->publisher || adopted->publisher == owner.token) {
            return;
        }
        // The request that published this point went on past it (its screenshot turn, its
        // output) and this request diverged there. Those private points can no longer serve
        // the chain; holding them is what fills Host with one dead endpoint per step.
        auto* publisher = find_owner(adopted->publisher);
        if (!publisher || publisher->active) { return; }
        std::vector<Handle> dead;
        for (const auto& point : publisher->points) {
            if (!program.valid_checkpoint(point.handle)) { continue; }
            const auto summary = program.checkpoint_metadata(point.handle);
            if (summary.frontier > reused && !summary.leased) { dead.push_back(point.handle); }
        }
        for (const auto handle : dead) {
            if (remove_point(program, *publisher, handle)) { ++lineage_stats_.retired_private; }
        }
    }

    // A deeper point of the same chain was just published from `parent_ordinal`. The parent was
    // adopted exactly once and is now behind the chain's head: keep the newest
    // `lineage_cap_` such points (for retries and rewinds) and retire older ones outright,
    // without the optional-capture admission comparison they would otherwise win forever.
    void supersede(Program& program, std::uint64_t parent_ordinal, std::uint64_t lineage) {
        auto* parent = find_shared(parent_ordinal);
        if (!parent || !parent->parent || parent->lineage != lineage || parent->adoptions > 1 ||
            parent->superseded) {
            return;
        }
        parent->superseded = ++clock_;
        // Behind the head it is only a retry fallback: first in line when space is short.
        parent->priority.reused = false;
        for (;;) {
            Shared* oldest    = nullptr;
            std::size_t count = 0;
            for (auto& entry : shared_) {
                if (entry.lineage != lineage || !entry.superseded) { continue; }
                ++count;
                if (!oldest || entry.superseded < oldest->superseded) { oldest = &entry; }
            }
            if (count <= lineage_cap_) { return; }
            // A point that cannot go now (leased, held by a waiting request) drops out of the
            // count and falls back to ordinary retention.
            if (!retire_shared(program, *oldest)) { oldest->superseded = 0; }
        }
    }

    // A later step of the chain has published, so points an earlier step published deeper than
    // where this request resumed, and that nobody resumed from (the end of its screenshot turn),
    // are dead: source selection takes the deepest matching point, so a deeper one this request
    // passed over does not match the chain any more. A shallower one (a session recap ahead of
    // the task) can still serve the next task and is kept. Without this the dead points would
    // keep the head priority they inherited for good.
    void retire_dead_siblings(Program& program, std::uint64_t lineage, OwnerToken publisher,
                              std::uint32_t resumed_at) {
        std::vector<std::uint64_t> dead;
        for (const auto& entry : shared_) {
            if (entry.lineage != lineage || entry.publisher == publisher || entry.adoptions ||
                entry.superseded || !program.valid_checkpoint(entry.handle) ||
                program.checkpoint_metadata(entry.handle).frontier <= resumed_at) {
                continue;
            }
            dead.push_back(entry.ordinal);
        }
        for (const auto ordinal : dead) {
            if (const auto* entry = find_shared(ordinal)) { (void)retire_shared(program, *entry); }
        }
    }

    bool retire_shared(Program& program, const Shared& entry) {
        const auto handle  = entry.handle;
        const auto ordinal = entry.ordinal;
        if (std::any_of(waiting_.begin(), waiting_.end(),
                        [&](const auto& waiting) { return contains(waiting.held, handle); })) {
            return false;
        }
        if (catalog_reference_count(handle) > 1) {
            // A private point at the same position keeps the contents; only the public entry goes.
            std::erase_if(shared_, [&](const auto& item) { return item.ordinal == ordinal; });
            ++lineage_stats_.retired_public;
            return true;
        }
        if (program.valid_checkpoint(handle)) {
            if (program.checkpoint_metadata(handle).leased || !program.release_checkpoint(handle)) {
                return false;
            }
        }
        forget(handle);
        ++lineage_stats_.retired_public;
        return true;
    }

    std::size_t catalog_reference_count(Handle handle) const {
        std::size_t count = 0;
        for (const auto& owner : owners_) {
            count += std::count_if(owner.points.begin(), owner.points.end(),
                                   [&](const auto& point) { return point.handle == handle; });
        }
        count += std::count_if(shared_.begin(), shared_.end(),
                               [&](const auto& entry) { return entry.handle == handle; });
        return count;
    }

    std::size_t reference_count(Handle handle) const {
        auto count = catalog_reference_count(handle);
        for (const auto& waiting : waiting_) { count += contains(waiting.held, handle); }
        return count;
    }

    bool references(Handle handle) const { return reference_count(handle) != 0; }

    bool remove_point(Program& program, Owner& owner, Handle handle) {
        if (reference_count(handle) == 1 && program.valid_checkpoint(handle) &&
            !program.release_checkpoint(handle)) {
            return false;
        }
        std::erase_if(owner.points, [&](const auto& point) { return point.handle == handle; });
        if (!catalog_reference_count(handle)) { index_.erase(handle); }
        return true;
    }

    void forget(Handle handle) {
        index_.erase(handle);
        for (auto& owner : owners_) {
            std::erase_if(owner.points, [&](const auto& point) { return point.handle == handle; });
        }
        std::erase_if(shared_, [&](const auto& entry) { return entry.handle == handle; });
        for (auto& waiting : waiting_) {
            if (std::erase(waiting.held, handle) && !waiting.binding) { ++waiting.revocations; }
        }
    }

    void release_unreferenced(Program& program, Handle handle) {
        if (!catalog_reference_count(handle)) { index_.erase(handle); }
        if (references(handle)) { return; }
        if (program.valid_checkpoint(handle) && !program.release_checkpoint(handle)) {
            throw std::logic_error("Native could not release an unreferenced checkpoint owner");
        }
    }

    void replace_points(Program& program, Owner& owner, std::span<const Handle> carried) {
        std::vector<Handle> old;
        for (const auto& point : owner.points) { old.push_back(point.handle); }
        owner.points.clear();
        for (const auto handle : carried) {
            if (!program.valid_checkpoint(handle)) { continue; }
            const auto summary = program.checkpoint_metadata(handle);
            if (summary.role == CheckpointRole::SharedPrefix) {
                throw std::logic_error("Native carried a public point as a private continuation");
            }
            owner.points.push_back({handle, summary.role});
            index_.insert(program, handle, summary.frontier);
        }
        for (const auto handle : old) { release_unreferenced(program, handle); }
    }

    void prune(Program& program) {
        std::vector<Handle> stale;
        const auto inspect = [&](Handle handle) {
            if (!program.valid_checkpoint(handle) && !contains(stale, handle)) {
                stale.push_back(handle);
            }
        };
        for (const auto& owner : owners_) {
            for (const auto& point : owner.points) { inspect(point.handle); }
        }
        for (const auto& shared : shared_) { inspect(shared.handle); }
        for (const auto& waiting : waiting_) {
            for (const auto handle : waiting.held) { inspect(handle); }
        }
        for (const auto handle : stale) { forget(handle); }
        std::erase_if(owners_,
                      [](const auto& owner) { return !owner.active && owner.points.empty(); });
    }

    CacheRetentionPriority priority_for(Handle handle) const {
        CacheRetentionPriority result;
        const auto include = [&](CacheRetentionPriority priority) {
            if (priority.reused && (!result.reused || priority.last_demand > result.last_demand)) {
                result = priority;
            }
        };
        for (const auto& owner : owners_) {
            if (std::any_of(owner.points.begin(), owner.points.end(),
                            [&](const auto& point) { return point.handle == handle; })) {
                include(owner.priority);
            }
        }
        for (const auto& shared : shared_) {
            if (shared.handle == handle) { include(shared.priority); }
        }
        return result;
    }

    std::vector<Handle> victims() const {
        struct Ranked {
            Handle handle;
            CacheRetentionPriority priority;
            std::uint64_t age;
        };

        std::vector<Ranked> ranked;
        const auto add = [&](Handle handle, CacheRetentionPriority priority, std::uint64_t age) {
            const auto it = std::find_if(ranked.begin(), ranked.end(),
                                         [&](const auto& entry) { return entry.handle == handle; });
            if (it == ranked.end()) {
                ranked.push_back({handle, priority, age});
            } else if ((priority.reused && !it->priority.reused) ||
                       (priority.reused == it->priority.reused &&
                        (priority.reused ? priority.last_demand > it->priority.last_demand
                                         : age > it->age))) {
                *it = {handle, priority, age};
            }
        };
        for (const auto& owner : owners_) {
            for (const auto& point : owner.points) {
                add(point.handle, owner.priority, owner.retained_at);
            }
        }
        for (const auto& shared : shared_) {
            add(shared.handle, shared.priority, shared.retained_at);
        }
        for (const auto& waiting : waiting_) {
            for (const auto handle : waiting.held) { add(handle, {}, waiting.request); }
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.priority.reused != b.priority.reused) { return !a.priority.reused; }
            return (a.priority.reused ? a.priority.last_demand : a.age) <
                   (b.priority.reused ? b.priority.last_demand : b.age);
        });
        std::vector<Handle> out;
        for (const auto& entry : ranked) { out.push_back(entry.handle); }
        return out;
    }

    [[nodiscard]] ReclaimCursor reclaim_order(Program& program) const {
        ReclaimCursor cursor;
        for (const auto handle : victims()) {
            const auto frontier = program.checkpoint_metadata(handle).frontier;
            cursor.victims.push_back({handle, priority_for(handle),
                                      demand_priority(program.checkpoint_key(handle, frontier))});
        }
        for (const auto& owner : owners_) {
            cursor.owners.push_back(
                {owner.token, owner.priority, owner.proven_frontier, points(owner.token)});
        }
        return cursor;
    }

    static std::uint64_t useful(ContextResourceUsage released, ContextResourceUsage shortage) {
        if (shortage.state_slots) { return std::min(released.state_slots, shortage.state_slots); }
        if (shortage.main_kv_pages) {
            return std::min(released.main_kv_pages, shortage.main_kv_pages);
        }
        if (shortage.backend_kv_pages) {
            return std::min(released.backend_kv_pages, shortage.backend_kv_pages);
        }
        return std::min(released.host_bytes, shortage.host_bytes);
    }

    [[nodiscard]] std::vector<Handle> surviving(Program& program,
                                                std::span<const Handle> removed) const {
        // Only advertised entries can replace a future cache lookup. Waiting-only
        // checkpoints remain physical holders, but are not a caller's lookup fallback.
        std::vector<Handle> kept;
        const auto removed_set = handle_set(removed);
        HandleSet seen;
        const auto include = [&](Handle handle) {
            if (!removed_set.contains(handle) && program.valid_checkpoint(handle) &&
                seen.insert(handle).second) {
                kept.push_back(handle);
            }
        };
        for (const auto& owner : owners_) {
            for (const auto& point : owner.points) { include(point.handle); }
        }
        for (const auto& shared : shared_) { include(shared.handle); }
        return kept;
    }

    using NativeReclaimPlan = decltype(std::declval<Program&>().plan_reclaim(
        std::span<const Handle>{}, std::span<const Handle>{}, ContextResourceUsage{}));

    struct ActionFacts {
        CacheRetentionPriority priority;
        std::size_t order = 0;
        std::optional<std::vector<Handle>> kept;
        std::optional<std::uint64_t> loss;
    };

    // Facts of one Host release set under the evaluation's fixed cursor. Every demotion
    // plan quotes the same Host candidates, so they are computed once per evaluation.
    struct CandidateQuote {
        std::size_t amount = 0;
        ActionFacts facts;
        std::optional<bool> outer_admitted;
    };

    // An admission's recovery gain with every retained point kept, and the frontier of each
    // source's deepest retained fallback. Removing a point at any other frontier leaves every
    // source's fallback, and therefore the gain, unchanged.
    struct AdmissionQuote {
        std::uint64_t gain = 0;
        std::unordered_set<std::uint32_t> fallback_frontiers;
    };

    using QuoteKey = std::vector<std::uint32_t>;

    // These facts live only until this synchronous evaluation mutates resources or yields.
    // The fixed victim cursor may outlive them; newly published points still count as fallback.
    struct Evaluation {
        const NativeReclaimPlan* native = nullptr;
        std::optional<std::vector<Handle>> retained;
        std::map<QuoteKey, CandidateQuote> candidates;
        std::map<QuoteKey, AdmissionQuote> admissions;
    };

    // Live handles are unique per slot index, so sorted indices name a handle set.
    static QuoteKey quote_key(std::span<const Handle> handles) {
        QuoteKey key;
        key.reserve(handles.size());
        for (const auto handle : handles) { key.push_back(handle.index); }
        std::sort(key.begin(), key.end());
        key.erase(std::unique(key.begin(), key.end()), key.end());
        return key;
    }

    [[nodiscard]] std::span<const Handle> retained(Program& program, Evaluation& evaluation) const {
        if (!evaluation.retained) { evaluation.retained = surviving(program, {}); }
        return *evaluation.retained;
    }

    [[nodiscard]] std::uint64_t recovery_loss(Program& program, std::span<const Handle> removed,
                                              std::span<const Handle> kept,
                                              const Evaluation& evaluation) const {
        if (removed.empty()) { return 0; }
        return evaluation.native ? evaluation.native->recovery_loss(removed, kept)
                                 : program.checkpoint_recovery_loss(removed, kept);
    }

    [[nodiscard]] std::span<const Handle> action_survivors(Program& program,
                                                           std::span<const Handle> sources,
                                                           ActionFacts& facts,
                                                           Evaluation& evaluation) const {
        if (!facts.kept) {
            facts.kept.emplace();
            const auto all     = retained(program, evaluation);
            const auto removed = handle_set(sources);
            facts.kept->reserve(all.size());
            for (const auto handle : all) {
                if (!removed.contains(handle)) { facts.kept->push_back(handle); }
            }
        }
        return *facts.kept;
    }

    [[nodiscard]] std::uint64_t action_loss(Program& program, std::span<const Handle> sources,
                                            ActionFacts& facts, Evaluation& evaluation) const {
        if (!facts.loss) {
            facts.loss = sources.empty()
                             ? 0
                             : recovery_loss(program, sources,
                                             action_survivors(program, sources, facts, evaluation),
                                             evaluation);
        }
        return *facts.loss;
    }

    [[nodiscard]] CacheRetentionPriority action_priority(Program& program,
                                                         std::span<const Handle> sources,
                                                         const ReclaimCursor& cursor,
                                                         const Evaluation& evaluation) const {
        CacheRetentionPriority priority;
        if (sources.empty()) { return priority; }
        const auto source_set = handle_set(sources);
        const auto include    = [&](CacheRetentionPriority candidate) {
            if (candidate.reused &&
                (!priority.reused || candidate.last_demand > priority.last_demand)) {
                priority = candidate;
            }
        };
        for (const auto& victim : cursor.victims) {
            if (!source_set.contains(victim.handle)) { continue; }
            const auto private_owner =
                std::any_of(cursor.owners.begin(), cursor.owners.end(), [&](const auto& owner) {
                    return contains(owner.points, victim.handle);
                });
            if (!private_owner) { include(victim.priority); }
        }
        for (const auto& proof : cursor.owners) {
            if (!proof.priority.reused) { continue; }
            const auto* owner = find_owner(proof.owner);
            bool affected     = false;
            bool covered      = owner != nullptr;
            for (const auto handle : proof.points) {
                if (!source_set.contains(handle)) { continue; }
                affected      = true;
                bool fallback = false;
                if (owner) {
                    const auto frontier = program.checkpoint_metadata(handle).frontier;
                    for (const auto& point : owner->points) {
                        if (source_set.contains(point.handle) ||
                            !program.valid_checkpoint(point.handle)) {
                            continue;
                        }
                        const auto candidate = program.checkpoint_metadata(point.handle).frontier;
                        if (candidate < proof.proven_frontier || candidate > frontier) { continue; }
                        const auto loss =
                            recovery_loss(program, {&handle, 1}, {&point.handle, 1}, evaluation);
                        if (loss == frontier - candidate) {
                            fallback = true;
                            break;
                        }
                    }
                }
                covered = covered && fallback;
            }
            // The hot grant protects already adopted progress. A later exact same-owner
            // checkpoint can cover that proof while an unproven suffix is reclaimed.
            if (affected && !covered) { include(proof.priority); }
        }
        return priority;
    }

    [[nodiscard]] ActionFacts action_facts(Program& program, std::span<const Handle> sources,
                                           const ReclaimCursor& cursor,
                                           const Evaluation& evaluation) const {
        ActionFacts facts{.priority = action_priority(program, sources, cursor, evaluation)};
        if (!sources.empty()) {
            const auto source_set = handle_set(sources);
            for (std::size_t i = 0; i < cursor.victims.size(); ++i) {
                if (source_set.contains(cursor.victims[i].handle)) { facts.order = i; }
            }
        }
        return facts;
    }

    // How many of the coldest Host release actions one reclaim choice ranks by value.
    static constexpr std::size_t kHostRankWindow = 16;

    struct ActionRank {
        CacheRetentionPriority priority;
        std::uint64_t loss    = 0;
        std::uint64_t units   = 0;
        std::size_t order     = 0;
        std::uint64_t waiting = 0;
    };

    [[nodiscard]] ActionRank rank_action(Program& program, std::span<const Handle> sources,
                                         ActionFacts& facts, std::uint64_t units,
                                         Evaluation& evaluation, bool preserving = false) const {
        return {.priority = facts.priority,
                .loss     = preserving ? 0 : action_loss(program, sources, facts, evaluation),
                .units    = units,
                .order    = facts.order,
                .waiting  = preserving ? 0 : waiting_ticket(sources)};
    }

    static bool less_rank(const ActionRank& left, const ActionRank& right) {
        if (bool(left.waiting) != bool(right.waiting)) { return !left.waiting; }
        if (left.waiting != right.waiting) { return left.waiting > right.waiting; }
        if (left.priority.reused != right.priority.reused) { return !left.priority.reused; }
        if (left.priority.reused && left.priority.last_demand != right.priority.last_demand) {
            return left.priority.last_demand < right.priority.last_demand;
        }
        if (!left.priority.reused) {
            const auto a = static_cast<unsigned __int128>(left.loss) * right.units;
            const auto b = static_cast<unsigned __int128>(right.loss) * left.units;
            if (a != b) { return a < b; }
        }
        return left.order < right.order;
    }

    [[nodiscard]] Admission preservation_admission(Program& program,
                                                   std::span<const Handle> sources,
                                                   const ReclaimCursor& cursor,
                                                   CacheRetentionPriority priority) const {
        Admission result{.priority = priority,
                         .sources  = std::vector<Handle>(sources.begin(), sources.end())};
        std::uint32_t deepest = 0;
        for (const auto handle : sources) {
            const auto frontier = program.checkpoint_metadata(handle).frontier;
            if (!result.checkpoint || frontier > deepest) {
                result.checkpoint = handle;
                deepest           = frontier;
            }
            const auto found =
                std::find_if(cursor.victims.begin(), cursor.victims.end(),
                             [&](const auto& victim) { return victim.handle == handle; });
            const auto demand =
                found == cursor.victims.end() ? CacheRetentionPriority{} : found->demand;
            if (demand.reused &&
                (!result.demand.reused || demand.last_demand > result.demand.last_demand)) {
                result.demand = demand;
            }
        }
        if (result.demand.reused) { result.priority = result.demand; }
        return result;
    }

    [[nodiscard]] std::uint64_t recovery_gain(Program& program, const Admission& admission,
                                              std::span<const Handle> kept,
                                              const Evaluation& evaluation = {}) const {
        if (admission.base) {
            std::uint32_t fallback = 0;
            for (const auto handle : kept) {
                fallback = std::max(fallback, program.checkpoint_recovery_frontier(
                                                  handle, *admission.base, admission.frontier));
            }
            return admission.frontier - fallback;
        }
        if (admission.checkpoint && program.valid_checkpoint(*admission.checkpoint)) {
            auto sources = admission.sources;
            if (sources.empty()) { sources.push_back(*admission.checkpoint); }
            auto other = std::vector<Handle>(kept.begin(), kept.end());
            std::erase_if(other, [&](auto handle) { return contains(sources, handle); });
            return recovery_loss(program, sources, other, evaluation);
        }
        return 0;
    }

    [[nodiscard]] bool admits(Program& program, std::span<const Handle> removed, ActionFacts& facts,
                              const std::optional<Admission>& admission,
                              const ReclaimCursor& cursor, Evaluation& evaluation,
                              bool preserving = false) const {
        if (!preserving && !may_revoke(removed, cursor.rights)) { return false; }
        if (!admission) { return true; }
        const auto victim = facts.priority;
        if (victim.reused && (!admission->priority.reused ||
                              victim.last_demand >= admission->priority.last_demand)) {
            return false;
        }
        const auto loss = preserving ? 0 : action_loss(program, removed, facts, evaluation);
        // This outer budget also covers Host victims of a preservation transfer. A hot
        // migration source cannot lend its permissions to the capture that requested space.
        if (cursor.gain_limit && (cursor.sacrificed >= *cursor.gain_limit ||
                                  loss >= *cursor.gain_limit - cursor.sacrificed)) {
            return false;
        }
        if (admission->demand.reused) { return true; }
        if (preserving) {
            return recovery_gain(program, *admission, retained(program, evaluation), evaluation) >
                   loss;
        }
        return removal_gain(program, *admission, removed, facts, evaluation) > loss;
    }

    // recovery_gain(admission, retained minus removed), answered from the admission's quote
    // unless a removed point sits at one of its sources' fallback frontiers.
    [[nodiscard]] std::uint64_t removal_gain(Program& program, const Admission& admission,
                                             std::span<const Handle> removed, ActionFacts& facts,
                                             Evaluation& evaluation) const {
        const auto exact = [&] {
            return recovery_gain(program, admission,
                                 action_survivors(program, removed, facts, evaluation), evaluation);
        };
        if (admission.base || !admission.checkpoint ||
            !program.valid_checkpoint(*admission.checkpoint)) {
            return exact();
        }
        const auto& quote = admission_quote(program, admission, evaluation);
        const bool touches_fallback =
            std::any_of(removed.begin(), removed.end(), [&](Handle handle) {
                return program.valid_checkpoint(handle) &&
                       quote.fallback_frontiers.contains(
                           program.checkpoint_metadata(handle).frontier);
            });
        if (touches_fallback) { return exact(); }
#ifdef NINFER_VERIFY_RECLAIM_QUOTES
        if (exact() != quote.gain) { throw std::logic_error("admission quote diverged"); }
#endif
        return quote.gain;
    }

    [[nodiscard]] const AdmissionQuote& admission_quote(Program& program,
                                                        const Admission& admission,
                                                        Evaluation& evaluation) const {
        auto sources = admission.sources;
        if (sources.empty()) { sources.push_back(*admission.checkpoint); }
        auto [entry, inserted] = evaluation.admissions.try_emplace(quote_key(sources));
        if (!inserted) { return entry->second; }
        const auto all = retained(program, evaluation);
        const auto source_set = handle_set(sources);
        std::vector<Handle> other;
        other.reserve(all.size());
        for (const auto handle : all) {
            if (!source_set.contains(handle)) { other.push_back(handle); }
        }
        auto& quote = entry->second;
        quote.gain  = recovery_loss(program, sources, other, evaluation);
        for (const auto source : sources) {
            if (!program.valid_checkpoint(source)) { continue; }
            // Survivors exclude every source exactly as in the combined quote, so this is the
            // source's own deepest fallback within that quote.
            const auto frontier = program.checkpoint_metadata(source).frontier;
            const auto lost     = recovery_loss(program, {&source, 1}, other, evaluation);
            if (lost < frontier) { quote.fallback_frontiers.insert(frontier - lost); }
        }
        return quote;
    }

    [[nodiscard]] CandidateQuote& candidate_quote(Program& program,
                                                  std::span<const Handle> combined,
                                                  const ReclaimCursor& cursor,
                                                  Evaluation& evaluation) const {
        auto [entry, inserted] = evaluation.candidates.try_emplace(quote_key(combined));
        if (inserted) {
            entry->second.amount = program.host_bytes_released(combined);
            entry->second.facts  = action_facts(program, combined, cursor, evaluation);
        }
#ifdef NINFER_VERIFY_RECLAIM_QUOTES
        else {
            auto fresh = action_facts(program, combined, cursor, evaluation);
            if (program.host_bytes_released(combined) != entry->second.amount ||
                fresh.order != entry->second.facts.order ||
                fresh.priority.reused != entry->second.facts.priority.reused ||
                fresh.priority.last_demand != entry->second.facts.priority.last_demand ||
                action_loss(program, combined, fresh, evaluation) !=
                    action_loss(program, combined, entry->second.facts, evaluation)) {
                throw std::logic_error("candidate quote diverged");
            }
        }
#endif
        return entry->second;
    }

    void commit_release(Program& program, std::span<const Handle> handles, ReclaimCursor& cursor,
                        std::optional<std::uint64_t> quoted_loss = std::nullopt) {
        if (handles.empty()) { return; }
        if (!may_revoke(handles, cursor.rights)) {
            throw std::logic_error("cache release lacks waiting-source revocation rights");
        }
        const auto loss =
            cursor.gain_limit
                ? (quoted_loss
                       ? *quoted_loss
                       : program.checkpoint_recovery_loss(handles, surviving(program, handles)))
                : 0;
        for (const auto handle : handles) {
            if (!erase(program, handle, cursor.rights)) {
                throw std::logic_error("quoted cache release changed before commit");
            }
        }
        cursor.sacrificed =
            loss > UINT64_MAX - cursor.sacrificed ? UINT64_MAX : cursor.sacrificed + loss;
    }

    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims_in_order(Program& program, std::size_t bytes, std::optional<Handle> incoming,
                          std::span<const Handle> later_snapshots, std::span<const Handle> excluded,
                          std::optional<Admission> admission, const ReclaimCursor& cursor,
                          Evaluation& evaluation, bool bytes_are_shortage = false) const {
        const auto usage    = program.physical_usage();
        const auto free     = usage.capacity.host_bytes - usage.occupied.host_bytes;
        const auto required = bytes_are_shortage ? bytes : (bytes > free ? bytes - free : 0);
        if (!required) { return std::vector<Handle>{}; }
        if (!bytes_are_shortage && bytes > usage.capacity.host_bytes) { return std::nullopt; }
        if (!admission && incoming) {
            admission = preservation_admission(
                program, {&*incoming, 1}, cursor,
                action_priority(program, {&*incoming, 1}, cursor, evaluation));
        }
        std::vector<Handle> allowed;
        for (const auto& victim : cursor.victims) {
            const auto handle = victim.handle;
            if (incoming != handle && !contains(excluded, handle) && references(handle) &&
                program.valid_checkpoint(handle)) {
                allowed.push_back(handle);
            }
        }
        // Paused snapshots are offered only by required recovery and keep their explicit
        // Engine order; ordinary optional writes cannot acquire these rights.
        if (!admission) {
            for (const auto handle : later_snapshots) {
                if (!contains(excluded, handle) && !contains(allowed, handle) &&
                    program.valid_checkpoint(handle)) {
                    allowed.push_back(handle);
                }
            }
        }
        auto actions = program.plan_releases(allowed, excluded, {.host_bytes = required});
        // Value ranking quotes recovery loss over every surviving point, so ranking every action
        // grows quadratically with the StateImages held on Host. Rank only a window of the
        // coldest actions instead. Cursor victims are ordered coldest first and an action is as
        // recent as its most recent source; paused snapshots follow ordinary history in the
        // Engine's explicit order. Held sources these rights may not revoke never enter the
        // window. When nothing in the window is admissible, the next coldest actions replace it.
        std::unordered_map<std::uint32_t, std::pair<Handle, std::size_t>> positions;
        for (std::size_t i = 0; i < cursor.victims.size(); ++i) {
            const auto handle = cursor.victims[i].handle;
            positions.insert_or_assign(handle.index, std::pair{handle, i});
        }
        struct Candidate {
            std::size_t snapshot;
            std::size_t recency;
            std::size_t action;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(actions.size());
        for (std::size_t i = 0; i < actions.size(); ++i) {
            if (!may_revoke(actions[i].sources, cursor.rights)) { continue; }
            std::size_t snapshot_order = 0;
            for (std::size_t snapshot = 0; snapshot < later_snapshots.size(); ++snapshot) {
                if (contains(actions[i].sources, later_snapshots[snapshot])) {
                    snapshot_order = snapshot + 1;
                }
            }
            std::size_t recency = 0;
            for (const auto handle : actions[i].sources) {
                const auto found = positions.find(handle.index);
                recency          = std::max(recency, found == positions.end() || found->second.first != handle
                                                         ? cursor.victims.size()
                                                         : found->second.second);
            }
            candidates.push_back({snapshot_order, recency, i});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            return std::tie(left.snapshot, left.recency, left.action) <
                   std::tie(right.snapshot, right.recency, right.action);
        });
        std::size_t next_candidate = 0;
        std::vector<std::size_t> window;
        const auto refill = [&] {
            while (window.size() < kHostRankWindow && next_candidate < candidates.size()) {
                window.push_back(candidates[next_candidate++].action);
            }
        };
        std::vector<Handle> selected;
        std::size_t released = 0;
        while (released < required) {
            refill();
            if (window.empty()) { return std::nullopt; }
            std::optional<std::size_t> best;
            std::vector<Handle> best_union;
            std::size_t best_bytes = 0;
            ActionRank best_rank;
            std::size_t best_snapshot_order = 0;
            for (std::size_t slot = 0; slot < window.size(); ++slot) {
                const auto i  = window[slot];
                auto combined = selected;
                for (const auto handle : actions[i].sources) {
                    if (!contains(combined, handle)) { combined.push_back(handle); }
                }
                auto& quote       = candidate_quote(program, combined, cursor, evaluation);
                const auto amount = quote.amount;
                if (amount <= released) { continue; }
                auto& facts = quote.facts;
                // A migration must satisfy both its own preservation rights and the
                // optional capture that requested space. Source heat cannot grant the
                // outer capture permission to erase unrelated hot Host history.
                if (!admits(program, combined, facts, admission, cursor, evaluation)) { continue; }
                if (!quote.outer_admitted) {
                    quote.outer_admitted =
                        admits(program, combined, facts, cursor.admission, cursor, evaluation);
                }
                if (!*quote.outer_admitted) { continue; }
                const auto rank            = rank_action(program, combined, facts,
                                                         std::min<std::size_t>(amount, required), evaluation);
                std::size_t snapshot_order = 0;
                for (std::size_t snapshot = 0; snapshot < later_snapshots.size(); ++snapshot) {
                    if (contains(combined, later_snapshots[snapshot])) {
                        snapshot_order = snapshot + 1;
                    }
                }
                // Ordinary history is exhausted before Engine-owned paused snapshots.
                // Snapshot candidates retain the Engine's explicit youngest-first order.
                if (!best || snapshot_order < best_snapshot_order ||
                    (snapshot_order == best_snapshot_order && less_rank(rank, best_rank))) {
                    best                = slot;
                    best_union          = std::move(combined);
                    best_bytes          = amount;
                    best_rank           = rank;
                    best_snapshot_order = snapshot_order;
                }
            }
            if (!best) {
                window.clear();
                continue;
            }
            selected = std::move(best_union);
            released = best_bytes;
            window.erase(window.begin() + static_cast<std::ptrdiff_t>(*best));
        }
        return selected;
    }

    void balance_reused(Program& program) {
        const auto amounts = [](ContextResourceUsage r) {
            return std::array<std::uint64_t, 4>{r.state_slots, r.main_kv_pages, r.backend_kv_pages,
                                                r.host_bytes};
        };

        struct ProtectedEntry {
            CacheRetentionPriority* priority;
            std::uint64_t* retained_at;
            std::uint64_t ordinal;
            std::vector<Handle> handles;
            std::optional<std::array<std::uint64_t, 4>> footprint;
        };

        std::vector<ProtectedEntry> entries;
        for (auto& owner : owners_) {
            if (!owner.priority.reused) { continue; }
            std::vector<Handle> handles;
            handles.reserve(owner.points.size());
            for (const auto& point : owner.points) { handles.push_back(point.handle); }
            entries.push_back(
                {&owner.priority, &owner.retained_at, owner.token, std::move(handles)});
        }
        for (auto& shared : shared_) {
            if (shared.priority.reused) {
                entries.push_back(
                    {&shared.priority, &shared.retained_at, shared.ordinal, {shared.handle}});
            }
        }
        if (entries.empty()) { return; }
        const auto capacities = amounts(program.physical_usage().capacity);
        std::vector<Handle> reused;
        for (;;) {
            reused.clear();
            for (const auto& entry : entries) {
                if (!entry.priority->reused) { continue; }
                for (const auto handle : entry.handles) {
                    if (!contains(reused, handle)) { reused.push_back(handle); }
                }
            }
            if (reused.empty()) { return; }
            // Shared physical objects still count once in the whole protected set.
            const auto usage       = amounts(program.checkpoint_footprint(reused));
            ProtectedEntry* demote = nullptr;
            for (std::size_t resource = 0; resource < usage.size() && !demote; ++resource) {
                if (usage[resource] <= capacities[resource] * 3ULL / 4ULL) { continue; }
                ProtectedEntry* newest = nullptr;
                ProtectedEntry* oldest = nullptr;
                for (auto& entry : entries) {
                    if (!entry.priority->reused) { continue; }
                    // This loop only changes protection priority, never Native contents.
                    if (!entry.footprint) {
                        entry.footprint = amounts(program.checkpoint_footprint(entry.handles));
                    }
                    if (!(*entry.footprint)[resource]) { continue; }
                    if (!newest || entry.priority->last_demand > newest->priority->last_demand ||
                        (entry.priority->last_demand == newest->priority->last_demand &&
                         entry.ordinal > newest->ordinal)) {
                        newest = &entry;
                    }
                    if (!oldest || entry.priority->last_demand < oldest->priority->last_demand ||
                        (entry.priority->last_demand == oldest->priority->last_demand &&
                         entry.ordinal < oldest->ordinal)) {
                        oldest = &entry;
                    }
                }
                if (oldest != newest) { demote = oldest; }
            }
            if (!demote) { return; }
            demote->priority->reused = false;
            *demote->retained_at     = ++clock_;
        }
    }

    bool enabled_;
    std::uint32_t lineage_cap_ = 0;
    double lrfu_lambda_        = 0;
    // Diagnostics only; counted from const admission queries too.
    mutable LineageStats lineage_stats_;
    ContextMachineCostModel costs_;
    PrefixIndex<Model> index_;
    std::vector<Owner> owners_;
    std::vector<Shared> shared_;
    std::vector<WaitingSource> waiting_;
    static constexpr std::size_t kDemandCapacity = 256;
    std::vector<Demand> demand_;
    std::uint64_t clock_   = 0;
    std::uint64_t ordinal_ = 0;
};
} // namespace ninfer::runtime
