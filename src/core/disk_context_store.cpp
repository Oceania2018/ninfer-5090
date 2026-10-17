#include "core/disk_context_store.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninfer {
namespace {

[[noreturn]] void throw_errno(int error, const std::string& what) {
    throw std::system_error(error, std::generic_category(), what);
}

} // namespace

DiskExtent::~DiskExtent() { (void)release(); }

DiskExtent::DiskExtent(DiskExtent&& other) noexcept
    : owner_(other.owner_), offset_(other.offset_), bytes_(other.bytes_) {
    other.disarm();
}

DiskExtent& DiskExtent::operator=(DiskExtent&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_  = other.owner_;
    offset_ = other.offset_;
    bytes_  = other.bytes_;
    other.disarm();
    return *this;
}

bool DiskExtent::release() noexcept {
    if (!valid()) { return false; }
    owner_->release(offset_, bytes_);
    disarm();
    return true;
}

void DiskExtent::disarm() noexcept {
    owner_  = nullptr;
    offset_ = 0;
    bytes_  = 0;
}

DiskContextStore::DiskContextStore(const std::filesystem::path& directory,
                                   std::size_t capacity_bytes)
    : directory_(directory), capacity_bytes_(capacity_bytes & ~(alignment - 1)) {
    if (capacity_bytes_ == 0) {
        throw std::invalid_argument("disk context capacity must be at least one 4 KiB block");
    }
    if (capacity_bytes_ > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
        throw std::invalid_argument("disk context capacity is out of range");
    }
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    if (error) {
        throw std::system_error(error, "disk context directory " + directory_.string() +
                                           " cannot be created");
    }
    fd_ = ::open(directory_.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    if (fd_ < 0) {
        throw_errno(errno, "disk context directory " + directory_.string() +
                               " does not support an unnamed file (O_TMPFILE)");
    }
    // posix_fallocate returns the error instead of setting errno.
    if (const int result = ::posix_fallocate(fd_, 0, static_cast<off_t>(capacity_bytes_));
        result != 0) {
        ::close(fd_);
        fd_ = -1;
        throw_errno(result, "disk context directory " + directory_.string() + " cannot hold " +
                                std::to_string(capacity_bytes_ >> 20) + " MiB");
    }
    free_extents_.reserve(capacity_bytes_ / alignment / 2 + 1);
    free_extents_.push_back({0, capacity_bytes_});
}

DiskContextStore::~DiskContextStore() {
    if (fd_ >= 0) { ::close(fd_); }
}

std::size_t DiskContextStore::allocation_bytes(std::size_t bytes) noexcept {
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return (bytes + alignment - 1) & ~(alignment - 1);
}

bool DiskContextStore::can_allocate(std::size_t bytes) const noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    if (rounded == 0) { return false; }
    for (const FreeExtent& extent : free_extents_) {
        if (extent.bytes >= rounded) { return true; }
    }
    return false;
}

std::optional<DiskExtent> DiskContextStore::allocate(std::size_t bytes) noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    if (rounded == 0) { return std::nullopt; }
    for (auto it = free_extents_.begin(); it != free_extents_.end(); ++it) {
        if (it->bytes < rounded) { continue; }
        const std::size_t offset = it->offset;
        it->offset += rounded;
        it->bytes -= rounded;
        if (it->bytes == 0) { free_extents_.erase(it); }
        occupied_bytes_ += rounded;
        return DiskExtent(*this, offset, rounded);
    }
    return std::nullopt;
}

void DiskContextStore::release(std::size_t offset, std::size_t bytes) noexcept {
    // Free extents stay sorted by offset and coalesced with both neighbours.
    auto next = free_extents_.begin();
    while (next != free_extents_.end() && next->offset < offset) { ++next; }
    const bool joins_previous =
        next != free_extents_.begin() && std::prev(next)->offset + std::prev(next)->bytes == offset;
    const bool joins_next = next != free_extents_.end() && offset + bytes == next->offset;
    if (joins_previous && joins_next) {
        std::prev(next)->bytes += bytes + next->bytes;
        free_extents_.erase(next);
    } else if (joins_previous) {
        std::prev(next)->bytes += bytes;
    } else if (joins_next) {
        next->offset = offset;
        next->bytes += bytes;
    } else {
        // Capacity was reserved for the worst-case alternating layout; insert cannot allocate.
        free_extents_.insert(next, FreeExtent{offset, bytes});
    }
    occupied_bytes_ -= bytes;
}

void DiskContextStore::check_range(const DiskExtent& extent, std::size_t extent_offset,
                                   std::size_t bytes) const {
    if (extent.owner_ != this || extent_offset > extent.bytes_ ||
        bytes > extent.bytes_ - extent_offset) {
        throw std::out_of_range("disk context I/O leaves its extent");
    }
}

void DiskContextStore::write(const DiskExtent& extent, std::size_t extent_offset,
                             std::span<const std::byte> bytes) const {
    check_range(extent, extent_offset, bytes.size());
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t written =
            ::pwrite(fd_, bytes.data() + done, bytes.size() - done,
                     static_cast<off_t>(extent.offset_ + extent_offset + done));
        if (written < 0) {
            if (errno == EINTR) { continue; }
            throw_errno(errno, "disk context write failed");
        }
        done += static_cast<std::size_t>(written);
    }
}

void DiskContextStore::read(const DiskExtent& extent, std::size_t extent_offset,
                            std::span<std::byte> bytes) const {
    check_range(extent, extent_offset, bytes.size());
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t read = ::pread(fd_, bytes.data() + done, bytes.size() - done,
                                     static_cast<off_t>(extent.offset_ + extent_offset + done));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            throw_errno(errno, "disk context read failed");
        }
        if (read == 0) { throw_errno(EIO, "disk context read ended inside its extent"); }
        done += static_cast<std::size_t>(read);
    }
}

void DiskContextStore::flush(const DiskExtent& extent) const {
    check_range(extent, 0, extent.bytes_);
    const auto offset = static_cast<off_t>(extent.offset_);
    const auto length = static_cast<off_t>(extent.bytes_);
    if (::sync_file_range(fd_, offset, length,
                          SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                              SYNC_FILE_RANGE_WAIT_AFTER) != 0) {
        throw_errno(errno, "disk context write-back failed");
    }
    // Advisory only: a failure leaves pages cached and is not a correctness problem.
    (void)::posix_fadvise(fd_, offset, length, POSIX_FADV_DONTNEED);
}

std::filesystem::path default_disk_context_directory() {
    if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache != nullptr && *cache != '\0') {
        return std::filesystem::path(cache) / "ninfer" / "disk-context";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / ".cache" / "ninfer" / "disk-context";
    }
    throw std::runtime_error(
        "--disk-context-dir is required when neither XDG_CACHE_HOME nor HOME is set");
}

} // namespace ninfer
