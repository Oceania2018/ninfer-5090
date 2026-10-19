#include "core/host_context_arena.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_reserve_publish_and_rollback() {
    ninfer::HostContextArena arena(1025, 1);
    expect(arena.capacity_bytes() == 1025 && arena.occupied_bytes() == 0,
           "idle fixed backing remains fully charged");
    auto source = arena.allocate(257);
    expect(source && source->bytes() == 512 && arena.reserved_bytes() == 512,
           "allocation includes alignment padding immediately");
    std::memset(source->data(), 0x51, source->bytes());
    source->publish();
    source->publish();
    expect(arena.live_bytes() == 512 && arena.reserved_bytes() == 0,
           "publication charges one unique live allocation");
    {
        auto target = arena.allocate(257);
        expect(target && target->data() != source->data() && arena.occupied_bytes() == 1024 &&
                   arena.live_bytes() == 512 && arena.reserved_bytes() == 512,
               "whole destination is reserved while the source remains resident");
        expect(!arena.allocate(1) && !arena.can_allocate(1),
               "unused trailing byte cannot satisfy aligned allocation");
    }
    expect(arena.occupied_bytes() == 512 && arena.reserved_bytes() == 0 &&
               arena.peak_occupied_bytes() == 1024,
           "cancelled destination rolls back occupancy without losing peak");
    expect(source->data()[0] == std::byte{0x51} && source->data()[511] == std::byte{0x51},
           "failed destination preserves the pinned source");
    auto moved = std::move(*source);
    expect(!source->valid() && arena.allocation_count() == 1,
           "moving ownership neither duplicates nor releases bytes");
    expect(moved.release() && !moved.release() && arena.occupied_bytes() == 0 &&
               arena.capacity_bytes() == 1025,
           "last owner release frees reusable bytes without hiding fixed backing");
    expect(!arena.allocate(0) && !arena.allocate(std::numeric_limits<std::size_t>::max()),
           "zero and overflowing requests fail without changing accounting");
}

void test_fragmentation_and_split() {
    ninfer::HostContextArena arena(2048, 256);
    auto full = arena.allocate(2048);
    full->publish();
    std::vector<ninfer::HostContextAllocation> pieces;
    pieces.reserve(8);
    while (full->bytes() > 256) {
        auto split = arena.split(std::move(*full), 256);
        pieces.push_back(std::move(split.first));
        *full = std::move(split.second);
    }
    pieces.push_back(std::move(*full));
    expect(arena.allocation_count() == 8 && arena.live_bytes() == 2048,
           "split reaches the smallest geometry without changing charged bytes");
    for (std::size_t index = 0; index < pieces.size(); index += 2) {
        (void)pieces[index].release();
    }
    expect(arena.free_bytes() == 1024 && !arena.can_allocate(512) && !arena.allocate(512),
           "actual contiguous extents, not total free bytes, govern availability");
    (void)pieces[1].release();
    auto merged = arena.allocate(768);
    expect(merged.has_value(), "adjacent released split pieces coalesce for another geometry");
    bool invalid_split_rejected = false;
    try {
        (void)arena.split(std::move(*merged), 1);
    } catch (const std::out_of_range&) { invalid_split_rejected = true; }
    expect(invalid_split_rejected && merged->valid() && merged->bytes() == 768,
           "invalid split leaves its source allocation owned");
    pieces.clear();
    merged.reset();
    expect(arena.occupied_bytes() == 0 && arena.allocation_count() == 0 && arena.can_allocate(2048),
           "all split pieces return to one complete reusable extent");
    ninfer::HostContextArena disabled(0, 256);
    expect(!disabled.allocate(256) && disabled.capacity_bytes() == 0,
           "Host zero has no backing or allocations");
}
// Images take the top and KV the bottom, so a released image leaves a hole the next image fits
// even while KV runs come and go between them.
void test_high_placement_keeps_images_apart_from_kv() {
    using ninfer::HostPlacement;
    ninfer::HostContextArena arena(4096, 256);
    auto image_a = arena.allocate(1024, HostPlacement::High);
    auto kv_a    = arena.allocate(256);
    auto image_b = arena.allocate(1024, HostPlacement::High);
    auto kv_b    = arena.allocate(512);
    expect(image_a && kv_a && image_b && kv_b, "mixed placements fit the backing");
    const auto* base = kv_a->data();
    expect(image_a->data() == base + 3072 && image_b->data() == base + 2048 &&
               kv_b->data() == base + 256,
           "images fill from the top and KV from the bottom");
    expect(arena.largest_free_bytes() == 1280 && arena.free_extent_count() == 1,
           "the free middle stays one extent");
    (void)kv_a->release();
    (void)image_a->release();
    expect(arena.free_extent_count() == 3 && arena.largest_free_bytes() == 1280,
           "released image and KV leave separate holes");
    auto kv_c = arena.allocate(256);
    expect(kv_c && kv_c->data() == base, "KV reuses the low hole first");
    auto image_c = arena.allocate(1024, HostPlacement::High);
    expect(image_c && image_c->data() == base + 3072,
           "the next image takes the released image's exact hole");
    auto image_d = arena.allocate(1024, HostPlacement::High);
    expect(image_d && image_d->data() == base + 1024 && arena.largest_free_bytes() == 256,
           "a full top falls back to the highest fitting extent");
    expect(!arena.allocate(1024, HostPlacement::High) && arena.free_bytes() == 256,
           "High placement still requires one contiguous extent");
}
} // namespace

int main() {
    int count               = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        test_reserve_publish_and_rollback();
        test_fragmentation_and_split();
        test_high_placement_keeps_images_apart_from_kv();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
