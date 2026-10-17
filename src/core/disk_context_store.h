#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

class DiskContextStore;

// A unique extent of the disk context file. Destruction returns the bytes to the store; the
// extent never outlives the store that issued it.
class DiskExtent {
public:
    DiskExtent() noexcept = default;
    ~DiskExtent();
    DiskExtent(const DiskExtent&)            = delete;
    DiskExtent& operator=(const DiskExtent&) = delete;
    DiskExtent(DiskExtent&& other) noexcept;
    DiskExtent& operator=(DiskExtent&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    bool release() noexcept;

private:
    friend class DiskContextStore;

    DiskExtent(DiskContextStore& owner, std::size_t offset, std::size_t bytes) noexcept
        : owner_(&owner), offset_(offset), bytes_(bytes) {}

    void disarm() noexcept;

    DiskContextStore* owner_ = nullptr;
    std::size_t offset_      = 0;
    std::size_t bytes_       = 0;
};

// One startup-fixed, process-private file that holds dehydrated context checkpoints. The file is
// created unnamed (O_TMPFILE), so it disappears with the process and never survives a restart;
// its whole capacity is preallocated so a full disk fails at startup, not during serving.
//
// Allocation and release belong to the Engine control thread. read/write are positional and may
// run on an I/O worker concurrently with allocation of other extents.
class DiskContextStore {
public:
    static constexpr std::size_t alignment = 4096;

    DiskContextStore(const std::filesystem::path& directory, std::size_t capacity_bytes);
    ~DiskContextStore();
    DiskContextStore(const DiskContextStore&)            = delete;
    DiskContextStore& operator=(const DiskContextStore&) = delete;
    DiskContextStore(DiskContextStore&&)                 = delete;
    DiskContextStore& operator=(DiskContextStore&&)      = delete;

    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] static std::size_t allocation_bytes(std::size_t bytes) noexcept;
    [[nodiscard]] bool can_allocate(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<DiskExtent> allocate(std::size_t bytes) noexcept;

    // Positional I/O inside one extent. Both throw std::system_error on an I/O failure and
    // std::out_of_range when the span leaves the extent.
    void write(const DiskExtent& extent, std::size_t extent_offset,
               std::span<const std::byte> bytes) const;
    void read(const DiskExtent& extent, std::size_t extent_offset, std::span<std::byte> bytes) const;

    // Writes back and drops the extent's page-cache pages so dehydrated content does not compete
    // with the pinned Host pool for RAM.
    void flush(const DiskExtent& extent) const;

private:
    friend class DiskExtent;

    struct FreeExtent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    void check_range(const DiskExtent& extent, std::size_t extent_offset,
                     std::size_t bytes) const;
    void release(std::size_t offset, std::size_t bytes) noexcept;

    std::filesystem::path directory_;
    int fd_                     = -1;
    std::size_t capacity_bytes_ = 0;
    std::size_t occupied_bytes_ = 0;
    std::vector<FreeExtent> free_extents_;
};

// Default directory when the operator enables the disk tier without naming one:
// $XDG_CACHE_HOME/ninfer/disk-context, else $HOME/.cache/ninfer/disk-context.
[[nodiscard]] std::filesystem::path default_disk_context_directory();

} // namespace ninfer
