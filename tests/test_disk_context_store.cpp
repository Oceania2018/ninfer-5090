#include "core/disk_context_store.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::filesystem::path scratch_directory() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ninfer_disk_context_store_test";
    std::filesystem::remove_all(path);
    return path;
}

void test_unnamed_file_and_round_trip() {
    const std::filesystem::path directory = scratch_directory() / "nested";
    ninfer::DiskContextStore store(directory, 64 * 1024 + 123);
    expect(std::filesystem::is_directory(directory), "missing directory is created");
    expect(std::filesystem::is_empty(directory), "backing file is unnamed");
    expect(store.capacity_bytes() == 64 * 1024 && store.occupied_bytes() == 0,
           "capacity rounds down to whole blocks");

    auto extent = store.allocate(5000);
    expect(extent && extent->bytes() == 8192 && store.occupied_bytes() == 8192,
           "allocation rounds up to whole blocks");
    std::vector<std::byte> written(5000);
    for (std::size_t i = 0; i < written.size(); ++i) {
        written[i] = static_cast<std::byte>(i * 7U + 3U);
    }
    store.write(*extent, 100, written);
    store.flush(*extent);
    std::vector<std::byte> read(5000);
    store.read(*extent, 100, read);
    expect(std::memcmp(written.data(), read.data(), written.size()) == 0,
           "positional read returns written bytes exactly");

    bool rejected = false;
    try {
        store.write(*extent, 8000, std::span<const std::byte>(written.data(), 500));
    } catch (const std::out_of_range&) { rejected = true; }
    expect(rejected, "I/O cannot leave its extent");

    expect(extent->release() && !extent->release() && store.occupied_bytes() == 0,
           "release returns bytes exactly once");
}

void test_first_fit_and_coalescing() {
    ninfer::DiskContextStore store(scratch_directory(), 4 * 4096);
    auto a = store.allocate(4096);
    auto b = store.allocate(4096);
    auto c = store.allocate(4096);
    auto d = store.allocate(4096);
    expect(a && b && c && d && !store.allocate(1) && !store.can_allocate(1),
           "full store refuses further allocation");
    b->release();
    d->release();
    expect(!store.can_allocate(2 * 4096), "separated holes do not satisfy a larger request");
    c->release();
    auto merged = store.allocate(3 * 4096);
    expect(merged && merged->offset() == 4096, "released neighbours coalesce for first fit");
    a->release();
    merged->release();
    expect(store.can_allocate(4 * 4096) && store.free_bytes() == store.capacity_bytes(),
           "full release restores one contiguous extent");

    auto moved_from = store.allocate(4096);
    ninfer::DiskExtent moved = std::move(*moved_from);
    expect(!moved_from->valid() && moved.valid() && store.occupied_bytes() == 4096,
           "moving an extent neither duplicates nor releases it");
}

void test_startup_failures() {
    bool zero = false;
    try {
        ninfer::DiskContextStore store(scratch_directory(), 100);
    } catch (const std::invalid_argument&) { zero = true; }
    expect(zero, "capacity below one block is rejected");

    bool full = false;
    try {
        // Larger than any test filesystem; fallocate must fail at startup, not later.
        ninfer::DiskContextStore store(scratch_directory(), std::size_t{1} << 60);
    } catch (const std::system_error&) { full = true; }
    expect(full, "insufficient free space fails at startup");
}

void test_default_directory() {
    ::setenv("XDG_CACHE_HOME", "/x/cache", 1);
    expect(ninfer::default_disk_context_directory() == "/x/cache/ninfer/disk-context",
           "XDG_CACHE_HOME selects the default directory");
    ::unsetenv("XDG_CACHE_HOME");
    ::setenv("HOME", "/x/home", 1);
    expect(ninfer::default_disk_context_directory() == "/x/home/.cache/ninfer/disk-context",
           "HOME is the fallback default directory");
}

} // namespace

int main() {
    test_unnamed_file_and_round_trip();
    test_first_fit_and_coalescing();
    test_startup_failures();
    test_default_directory();
    std::filesystem::remove_all(std::filesystem::temp_directory_path() /
                                "ninfer_disk_context_store_test");
    if (failures != 0) { std::cerr << failures << " failure(s)\n"; }
    return failures == 0 ? 0 : 1;
}
