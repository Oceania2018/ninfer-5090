#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"

#include <algorithm>
#include <chrono>
#include <new>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

// Disk tier: a write-behind copy of cold Host-resident checkpoints. Host retention is unchanged;
// once Host deletes a checkpoint, its disk copy can still be rebuilt into Host as a shared prefix.
// See docs/maintainer/resource-scheduling-and-context-cache.md (Disk tier).

namespace ninfer::models::qwen3_5::detail {
namespace {

// One background copy moves at most this much, so a write finishes well before Host pressure
// that triggered it turns into deletion of the copied points.
constexpr std::size_t kDiskWriteJobBytes = std::size_t{1} << 30;

std::uint32_t page_columns(std::uint32_t coverage, std::uint32_t page) noexcept {
    const auto begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
    return std::min(static_cast<std::uint32_t>(kPagedKVPageSize), coverage - begin);
}

bool same_slot(const qwen3_5::HostStateSlotHandle& a,
               const qwen3_5::HostStateSlotHandle& b) noexcept {
    return a.owner == b.owner && a.index == b.index && a.generation == b.generation;
}

bool same_replica(const HostKVPageReplica& a, const HostKVPageReplica& b) noexcept {
    return a.extent == b.extent && a.page_offset == b.page_offset &&
           a.membership_node == b.membership_node && a.content_epoch == b.content_epoch &&
           a.committed_columns == b.committed_columns;
}

// Every page up to `coverage` has a current Host replica holding at least the restored columns.
bool host_complete(const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                   KVAddressSpaceHandle address, std::uint32_t coverage) {
    const auto count = kv_pages_for_frontier(coverage);
    if (addresses.committed_frontier(address) < coverage || addresses.mapped_pages(address) < count) {
        return false;
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto page = addresses.logical_page(address, index);
        if (!pages.host_replica_current(page) ||
            pages.committed_columns(page) < page_columns(coverage, index)) {
            return false;
        }
    }
    return true;
}

// Backend pages a restored point needs: MTP reads one token behind the main frontier.
std::uint32_t backend_coverage(const ProgramImpl& program, const CheckpointState& record) {
    const auto rule = program.speculative_backend == SpeculativeBackend::Mtp && record.frontier
                          ? record.frontier - 1U
                          : record.backend_frontier;
    return std::max(rule, record.backend_frontier);
}

} // namespace

bool ProgramImpl::start_disk_write(std::span<const CheckpointHandle> coldest) {
    if (!disk_store || !disk_io || disk_job || !host_state_images || !host_kv_extents) {
        return false;
    }
    const std::size_t image_bytes = state_images->host_layout().image_bytes;

    struct Selected {
        CheckpointHandle handle;
        std::uint32_t backend = 0;
    };

    struct HistoryCopy {
        std::uint32_t main_coverage    = 0;
        std::uint32_t backend_coverage = 0;
        std::shared_ptr<DiskBlob> main;
        std::shared_ptr<DiskBlob> backend;
    };

    const auto on_disk = [&](const PrefixShortlistKey& key) {
        return std::any_of(disk_records.begin(), disk_records.end(),
                           [&](const auto& record) { return record.key == key; });
    };

    // Pass 1: the coldest eligible points, deduplicated by semantic key, within one job budget.
    std::vector<Selected> selected;
    std::unordered_map<const KVHistory*, HistoryCopy> histories;
    std::size_t estimate = 0;
    for (const auto handle : coldest) {
        if (estimate >= kDiskWriteJobBytes) { break; }
        if (!valid_checkpoint(handle)) { continue; }
        const auto& record = checkpoint(handle);
        if (!record.identity || on_disk(record.key) ||
            std::any_of(selected.begin(), selected.end(), [&](const auto& prior) {
                return checkpoint(prior.handle).key == record.key;
            })) {
            continue;
        }
        if (!state_store->host_resident(record.state) ||
            !host_complete(*text_kv_addresses, *text_kv_pages, record.kv->text, record.frontier)) {
            continue;
        }
        const auto backend = record.kv->backend ? backend_coverage(*this, record) : 0U;
        if (record.kv->backend && !host_complete(*backend_kv_addresses, *backend_kv_pages,
                                                 *record.kv->backend, backend)) {
            continue;
        }
        auto& history = histories[record.kv.get()];
        const auto added_main =
            kv_pages_for_frontier(std::max(history.main_coverage, record.frontier)) -
            kv_pages_for_frontier(history.main_coverage);
        const auto added_backend =
            kv_pages_for_frontier(std::max(history.backend_coverage, backend)) -
            kv_pages_for_frontier(history.backend_coverage);
        history.main_coverage    = std::max(history.main_coverage, record.frontier);
        history.backend_coverage = std::max(history.backend_coverage, backend);
        estimate += image_bytes + added_main * text_host_kv_page_stride +
                    added_backend * backend_host_kv_page_stride;
        selected.push_back({handle, backend});
    }
    if (selected.empty()) { return false; }

    const auto allocate = [&](std::size_t bytes) -> std::shared_ptr<DiskBlob> {
        if (bytes == 0) { return nullptr; }
        // Least recently used records give way; extents shared by a surviving record stay.
        while (!disk_store->can_allocate(bytes) && !disk_records.empty()) {
            const auto victim = std::min_element(
                disk_records.begin(), disk_records.end(),
                [](const auto& a, const auto& b) { return a.last_used < b.last_used; });
            disk_records.erase(victim);
            ++disk_counters.evictions;
        }
        auto extent = disk_store->allocate(bytes);
        if (!extent) { return nullptr; }
        return std::make_shared<DiskBlob>(DiskBlob{std::move(*extent)});
    };

    // Pass 2: one KV blob per history, sized for its deepest selected point.
    DiskWriteJob job;
    std::vector<DiskWriteJob::Segment> segments;
    const auto add_segment = [&](const std::byte* source, std::size_t bytes, const DiskBlob& blob,
                                 std::size_t offset) {
        if (!segments.empty()) {
            auto& last = segments.back();
            if (last.blob == &blob && last.source + last.bytes == source &&
                last.offset + last.bytes == offset) {
                last.bytes += bytes;
                return;
            }
        }
        segments.push_back({source, bytes, &blob, offset});
    };
    const auto copy_kv = [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                             KVAddressSpaceHandle address, std::uint32_t coverage,
                             std::size_t stride, const DiskBlob& blob) {
        for (std::uint32_t index = 0; index < kv_pages_for_frontier(coverage); ++index) {
            const auto page    = addresses.logical_page(address, index);
            const auto replica = pages.host_replica(page);
            const auto view    = host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
            if (view.layout().page_stride != stride) {
                throw std::logic_error("disk tier Host KV stride disagrees with its pool");
            }
            add_segment(view.data(), stride, blob, static_cast<std::size_t>(index) * stride);
            job.pages.push_back({&pages, page, replica});
        }
    };
    std::unordered_map<const KVHistory*, bool> copied;
    for (auto& [history, copy] : histories) {
        const HistoryCopy& plan = copy;
        copy.main = allocate(kv_pages_for_frontier(plan.main_coverage) * text_host_kv_page_stride);
        if (!copy.main) { continue; }
        if (history->backend && plan.backend_coverage) {
            copy.backend =
                allocate(kv_pages_for_frontier(plan.backend_coverage) * backend_host_kv_page_stride);
            if (!copy.backend) {
                copy.main.reset();
                continue;
            }
        }
        copy_kv(*text_kv_addresses, *text_kv_pages, history->text, plan.main_coverage,
                text_host_kv_page_stride, *copy.main);
        if (copy.backend) {
            copy_kv(*backend_kv_addresses, *backend_kv_pages, *history->backend,
                    plan.backend_coverage, backend_host_kv_page_stride, *copy.backend);
        }
        job.bytes += copy.main->extent.bytes() + (copy.backend ? copy.backend->extent.bytes() : 0);
        copied[history] = true;
    }
    for (const auto& item : selected) {
        const auto& record = checkpoint(item.handle);
        const auto& copy   = histories[record.kv.get()];
        if (!copied.contains(record.kv.get())) { continue; }
        auto state = allocate(image_bytes);
        if (!state) { continue; }
        const auto slot = *state_store->host_slot(record.state);
        add_segment(host_state_images->view(slot).data, image_bytes, *state, 0);
        job.states.push_back({record.state, slot});
        job.bytes += state->extent.bytes();

        DiskCheckpointRecord out{.identity          = record.identity,
                                 .key               = record.key,
                                 .frontier          = record.frontier,
                                 .backend_frontier  = record.backend_frontier,
                                 .rope_delta        = record.rope_delta,
                                 .tail_hidden_valid = record.tail_hidden_valid,
                                 .state             = std::move(state)};
        out.main.blob     = copy.main;
        out.main.coverage = record.frontier;
        for (std::uint32_t page = 0; page < kv_pages_for_frontier(record.frontier); ++page) {
            out.main.columns.push_back(page_columns(record.frontier, page));
        }
        if (record.kv->backend) {
            DiskKVCopy backend{.blob = item.backend ? copy.backend : nullptr,
                               .coverage = item.backend};
            for (std::uint32_t page = 0; page < kv_pages_for_frontier(item.backend); ++page) {
                backend.columns.push_back(page_columns(item.backend, page));
            }
            out.backend = std::move(backend);
        }
        job.records.push_back(std::move(out));
    }
    if (job.records.empty()) { return false; }

    std::vector<const DiskBlob*> blobs;
    for (const auto& segment : segments) {
        if (std::find(blobs.begin(), blobs.end(), segment.blob) == blobs.end()) {
            blobs.push_back(segment.blob);
        }
    }
    // The worker only reads Host replicas and writes extents this job owns; the records keep both
    // the extents and (through the Program) the store alive until poll_disk_write retires the job.
    job.done = disk_io->submit([store = disk_store.get(), segments = std::move(segments),
                                blobs = std::move(blobs)] {
        for (const auto& segment : segments) {
            store->write(segment.blob->extent, segment.offset,
                         std::span<const std::byte>(segment.source, segment.bytes));
        }
        for (const auto* blob : blobs) { store->flush(blob->extent); }
    });
    disk_job = std::move(job);
    return true;
}

bool ProgramImpl::poll_disk_write() noexcept {
    if (!disk_job || disk_job->done.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return false;
    }
    DiskWriteJob job = std::move(*disk_job);
    disk_job.reset();
    bool current = true;
    try {
        job.done.get();
    } catch (...) {
        ++disk_counters.io_errors;
        current = false;
    }
    try {
        // A copied replica that was released or replaced during the copy may have been overwritten.
        for (const auto& token : job.states) {
            current = current && state_store->valid(token.state) &&
                      state_store->host_slot(token.state) &&
                      same_slot(*state_store->host_slot(token.state), token.slot);
        }
        for (const auto& token : job.pages) {
            current = current && token.pages->valid(token.page) &&
                      token.pages->host_resident(token.page) &&
                      same_replica(token.pages->host_replica(token.page), token.replica);
        }
        if (!current) {
            ++disk_counters.writes_discarded;
            return true;
        }
        for (auto& record : job.records) {
            record.last_used = ++disk_clock;
            disk_records.push_back(std::move(record));
        }
        ++disk_counters.writes;
        disk_counters.write_bytes += job.bytes;
    } catch (...) { ++disk_counters.writes_discarded; }
    return true;
}

std::optional<DiskMatch> ProgramImpl::disk_match(const RequestBasePlan& base) const {
    if (!disk_store || !base.impl_ || !base.impl_->allow_prefix_reuse) { return std::nullopt; }
    const auto prompt_tokens = base.summary().prompt_tokens;
    std::optional<DiskMatch> best;
    for (std::uint32_t index = 0; index < disk_records.size(); ++index) {
        const auto& record = disk_records[index];
        if (record.frontier > prompt_tokens || (best && record.frontier <= best->frontier) ||
            record.key.identity_tag != base.impl_->prefix_identity_tag || !record.identity ||
            base.impl_->prefix_digests.at(record.frontier) != record.key.digests ||
            !prefix_matches(*base.impl_->prompt, record.identity->ledger,
                            record.identity->prefix_identity, record.frontier)) {
            continue;
        }
        std::size_t bytes = state_images->host_layout().image_bytes +
                            record.main.columns.size() * text_host_kv_page_stride;
        auto pages = static_cast<std::uint32_t>(record.main.columns.size());
        if (record.backend) {
            bytes += record.backend->columns.size() * backend_host_kv_page_stride;
            pages += static_cast<std::uint32_t>(record.backend->columns.size());
        }
        best = DiskMatch{
            .record = index, .frontier = record.frontier, .host_bytes = bytes, .kv_pages = pages};
    }
    if (!best) { return best; }
    // Same remaining-work definition as inspect_source, so the two compare on one cost scale.
    best->remaining_work = runtime::make_prefill_work(best->frontier, prompt_tokens - best->frontier,
                                                      0, 0, prefill_chunk);
    for (const auto& item : base.impl_->prompt->vision_items) {
        if (std::any_of(item.token_spans.begin(), item.token_spans.end(), [&](const auto& span) {
                return span.begin + span.count > best->frontier;
            })) {
            ++best->remaining_work.vision_items;
            best->remaining_work.vision_patches += item.patch_count;
        }
    }
    return best;
}

// Phase 1 of a restore, on the control thread: every Host destination and descriptor the record
// needs is reserved here, so the reader thread only fills bytes nobody else can reach.
bool ProgramImpl::start_disk_restore(const DiskMatch& match) {
    if (!disk_store || !disk_reader || disk_restore || context_transaction_ ||
        !host_state_images || !host_kv_extents || match.record >= disk_records.size() ||
        disk_records[match.record].frontier != match.frontier) {
        return false;
    }
    DiskRestoreJob job{.record = disk_records[match.record], .host_bytes = match.host_bytes};
    const auto handle = reserve_checkpoint();
    if (!handle) { return false; }
    job.checkpoint = *handle;
    std::vector<DiskRestoreJob::Segment> segments;

    // Pages get their history's one address reference now; Host extents are reserved in the
    // largest runs the arena can place, halving on fragmentation like a demotion does.
    const auto reserve_pages = [&](LogicalKVPageStore& pages, std::size_t stride,
                                   const DiskKVCopy& copy, DiskRestoreJob::Pool& pool) {
        for (const auto columns : copy.columns) {
            pool.pages.push_back(pages.materialize_host_restore(columns));
        }
        std::size_t begin = 0;
        auto run          = pool.pages.size();
        while (begin < pool.pages.size()) {
            const auto count = std::min(run, pool.pages.size() - begin);
            auto extent =
                host_kv_extents->prepare_restore(pages, std::span(pool.pages).subspan(begin, count));
            if (!extent) {
                if (count == 1) { throw std::bad_alloc(); }
                run = count / 2;
                continue;
            }
            const auto view = host_kv_extents->writable_view(*extent);
            if (view.layout().page_stride != stride) {
                throw std::logic_error("disk tier Host KV stride disagrees with its pool");
            }
            segments.push_back({copy.blob.get(), begin * stride, view.data(), count * stride});
            pool.extents.push_back(std::move(*extent));
            begin += count;
        }
    };
    try {
        const auto image_bytes = state_images->host_layout().image_bytes;
        job.slot               = host_state_images->allocate();
        if (!job.slot) { throw std::bad_alloc(); }
        segments.push_back({job.record.state.get(), 0,
                            host_state_images->writable_view(*job.slot).data, image_bytes});
        reserve_pages(*text_kv_pages, text_host_kv_page_stride, job.record.main, job.main);
        if (job.record.backend) {
            if (!backend_kv_pages) { throw std::logic_error("disk record has an unknown backend"); }
            reserve_pages(*backend_kv_pages, backend_host_kv_page_stride, *job.record.backend,
                          job.backend);
        }
    } catch (const std::exception&) {
        unwind_disk_restore(job);
        return false;
    }
    // The job's record copy keeps every source extent alive even if the record is evicted.
    job.done = disk_reader->submit([store = disk_store.get(), segments = std::move(segments)] {
        for (const auto& segment : segments) {
            store->read(segment.blob->extent, segment.offset,
                        std::span<std::byte>(segment.destination, segment.bytes));
        }
    });
    disk_restore = std::move(job);
    return true;
}

// Phase 2, on the control thread once the read finished: publish the bytes and install a
// Host-only SharedPrefix checkpoint over its own history.
DiskRestoreProgress ProgramImpl::poll_disk_restore() noexcept {
    if (!disk_restore || context_transaction_ ||
        disk_restore->done.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return {};
    }
    DiskRestoreJob job = std::move(*disk_restore);
    disk_restore.reset();
    try {
        job.done.get();
    } catch (...) {
        ++disk_counters.io_errors;
        unwind_disk_restore(job);
        return {.finished = true};
    }
    try {
        if (!host_state_images->publish(*job.slot)) {
            throw std::logic_error("restored StateImage slot is not publishable");
        }
        const auto slot = *job.slot;
        job.slot.reset();
        job.state = state_store->install_host_checkpoint(slot);
        if (!job.state) {
            (void)host_state_images->release(slot);
            throw std::bad_alloc();
        }
        state_store->retain_checkpoint_reference(*job.state);
        job.state_referenced = true;
        const auto install = [&](KVAddressSpaceStore& addresses, DiskRestoreJob::Pool& pool,
                                 std::uint32_t coverage) {
            for (auto& extent : pool.extents) { (void)host_kv_extents->publish(std::move(extent)); }
            pool.extents.clear();
            const auto address = addresses.create_inactive();
            if (!address) { throw std::bad_alloc(); }
            try {
                addresses.install_inactive(*address, pool.pages, coverage);
            } catch (...) {
                (void)addresses.release(*address);
                throw;
            }
            pool.pages.clear();
            return *address;
        };
        job.text = install(*text_kv_addresses, job.main, job.record.main.coverage);
        if (job.record.backend) {
            job.backend_address =
                install(*backend_kv_addresses, job.backend, job.record.backend->coverage);
        }
    } catch (const std::exception&) {
        unwind_disk_restore(job);
        return {.finished = true};
    }

    auto history     = std::make_shared<KVHistory>();
    history->text    = *job.text;
    history->backend = job.backend_address;
    history->owner   = this;
    auto& slot       = checkpoints[job.checkpoint.index];
    slot.value       = CheckpointState{.kv                = history,
                                       .role              = runtime::CheckpointRole::SharedPrefix,
                                       .state             = *job.state,
                                       .identity          = job.record.identity,
                                       .key               = job.record.key,
                                       .frontier          = job.record.frontier,
                                       .backend_frontier  = job.record.backend_frontier,
                                       .rope_delta        = job.record.rope_delta,
                                       .tail_hidden_valid = job.record.tail_hidden_valid};
    slot.reserved    = false;
    try {
        refresh_history_requirements(history);
    } catch (...) { std::terminate(); }
    for (auto& record : disk_records) {
        if (record.state == job.record.state) { record.last_used = ++disk_clock; }
    }
    ++disk_counters.restores;
    disk_counters.restore_bytes += job.host_bytes;
    return {.finished = true, .checkpoint = job.checkpoint};
}

void ProgramImpl::abort_disk_restore() noexcept {
    if (!disk_restore) { return; }
    // The reader writes into reserved Host bytes; they stay reserved until it stops.
    try {
        disk_restore->done.wait();
    } catch (...) {}
    DiskRestoreJob job = std::move(*disk_restore);
    disk_restore.reset();
    unwind_disk_restore(job);
}

void ProgramImpl::unwind_disk_restore(DiskRestoreJob& job) noexcept {
    try {
        // Unpublished extents pin their pages; abort them before dropping the page references.
        for (auto* pool : {&job.main, &job.backend}) {
            for (auto& extent : pool->extents) { host_kv_extents->abort(extent); }
            pool->extents.clear();
        }
        for (const auto page : job.main.pages) {
            (void)text_kv_pages->release_reference(page, false);
        }
        for (const auto page : job.backend.pages) {
            (void)backend_kv_pages->release_reference(page, false);
        }
        if (job.backend_address) { (void)backend_kv_addresses->release(*job.backend_address); }
        if (job.text) { (void)text_kv_addresses->release(*job.text); }
        (void)host_kv_extents->release_unreferenced();
        if (job.state) {
            if (job.state_referenced) {
                (void)state_store->release_checkpoint_owner(*job.state);
            } else {
                (void)state_store->release(*job.state);
            }
        } else if (job.slot) {
            (void)host_state_images->release(*job.slot);
        }
        checkpoints[job.checkpoint.index].reserved = false;
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::qwen3_5::detail
