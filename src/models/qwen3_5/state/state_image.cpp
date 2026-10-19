#include "models/qwen3_5/state/state_image.h"

#include "core/device.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/nvfp4_slice_codec.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kStateImageAlignment = 256;

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(label);
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

std::uint32_t padded_dflash_capacity(std::uint32_t capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("StateImage DFlash capacity must be positive");
    }
    constexpr std::uint64_t alignment = 128;
    const std::uint64_t padded =
        (static_cast<std::uint64_t>(capacity) + alignment - 1U) & ~(alignment - 1U);
    if (padded > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("StateImage DFlash padded capacity exceeds int32");
    }
    return static_cast<std::uint32_t>(padded);
}

bool same_linear_spec(const LinearAttentionStatePoolSpec& left,
                      const LinearAttentionStatePoolSpec& right) noexcept {
    return left.layers == right.layers && left.conv_channels == right.conv_channels &&
           left.conv_width == right.conv_width && left.value_heads == right.value_heads &&
           left.value_head_dim == right.value_head_dim && left.key_head_dim == right.key_head_dim &&
           left.slot_count == right.slot_count && left.conv_dtype == right.conv_dtype;
}

bool same_dflash_spec(const std::optional<DFlashLocalStateSpec>& left,
                      const std::optional<DFlashLocalStateSpec>& right) noexcept {
    if (left.has_value() != right.has_value()) { return false; }
    if (!left) { return true; }
    return left->layers == right->layers && left->capacity == right->capacity &&
           left->kv_heads == right->kv_heads && left->head_dim == right->head_dim;
}

bool same_region(const LayoutRegion& left, const LayoutRegion& right) noexcept {
    return left.offset == right.offset && left.bytes == right.bytes &&
           left.alignment == right.alignment;
}

bool same_optional_region(const std::optional<LayoutRegion>& left,
                          const std::optional<LayoutRegion>& right) noexcept {
    return left.has_value() == right.has_value() && (!left || same_region(*left, *right));
}

bool same_recurrent_encoding(const StateImageRecurrentEncoding& left,
                             const StateImageRecurrentEncoding& right) noexcept {
    return left.values_bytes == right.values_bytes &&
           left.group_scale_offset == right.group_scale_offset &&
           left.group_scale_bytes == right.group_scale_bytes &&
           left.slice_scale_offset == right.slice_scale_offset &&
           left.slice_scale_bytes == right.slice_scale_bytes;
}

constexpr std::size_t kRecurrentEncodingAlignment = 16;

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    return checked_add(value, alignment - 1U, label) & ~(alignment - 1U);
}

// One layer's recurrent slot in its Host encoding; returns the encoded layer stride.
std::size_t plan_recurrent_encoding(const LinearAttentionStatePoolSpec& linear,
                                    HostStateStorage storage, StateImageRecurrentEncoding& out) {
    const std::size_t head_elements = checked_mul(static_cast<std::size_t>(linear.key_head_dim),
                                                  static_cast<std::size_t>(linear.value_head_dim),
                                                  "StateImage recurrent head overflow");
    const std::size_t elements = checked_mul(head_elements,
                                             static_cast<std::size_t>(linear.value_heads),
                                             "StateImage recurrent layer overflow");
    out = {};
    switch (storage) {
    case HostStateStorage::Fp32:
        out.values_bytes = checked_mul(elements, sizeof(float), "StateImage recurrent overflow");
        return out.values_bytes;
    case HostStateStorage::BFloat16:
        out.values_bytes = checked_mul(elements, 2U, "StateImage recurrent overflow");
        return align_up(out.values_bytes, kRecurrentEncodingAlignment,
                        "StateImage recurrent overflow");
    case HostStateStorage::Nvfp4Group16: {
        constexpr auto group = static_cast<std::size_t>(ops::kNvfp4SliceGroup);
        if (head_elements % group != 0) {
            throw std::invalid_argument(
                "StateImage NVFP4 host state needs value heads that are multiples of 16 values");
        }
        out.values_bytes       = elements / 2U;
        out.group_scale_offset = align_up(out.values_bytes, kRecurrentEncodingAlignment,
                                          "StateImage recurrent overflow");
        out.group_scale_bytes  = elements / group;
        out.slice_scale_offset =
            align_up(checked_add(out.group_scale_offset, out.group_scale_bytes,
                                 "StateImage recurrent overflow"),
                     kRecurrentEncodingAlignment, "StateImage recurrent overflow");
        out.slice_scale_bytes =
            checked_mul(static_cast<std::size_t>(linear.value_heads), sizeof(float),
                        "StateImage recurrent overflow");
        return align_up(checked_add(out.slice_scale_offset, out.slice_scale_bytes,
                                    "StateImage recurrent overflow"),
                        kRecurrentEncodingAlignment, "StateImage recurrent overflow");
    }
    }
    throw std::invalid_argument("StateImage host recurrent storage is invalid");
}

bool same_host_layout(const StateImageHostLayout& left,
                      const StateImageHostLayout& right) noexcept {
    return same_linear_spec(left.spec.linear, right.spec.linear) &&
           left.spec.hidden == right.spec.hidden &&
           same_dflash_spec(left.spec.dflash_local, right.spec.dflash_local) &&
           same_region(left.linear_conv, right.linear_conv) &&
           left.linear_conv_layer_bytes == right.linear_conv_layer_bytes &&
           same_region(left.linear_recurrent, right.linear_recurrent) &&
           left.linear_recurrent_layer_bytes == right.linear_recurrent_layer_bytes &&
           left.spec.host_recurrent == right.spec.host_recurrent &&
           same_recurrent_encoding(left.recurrent_encoding, right.recurrent_encoding) &&
           same_region(left.continuation_hidden, right.continuation_hidden) &&
           same_optional_region(left.dflash_local_k, right.dflash_local_k) &&
           same_optional_region(left.dflash_local_v, right.dflash_local_v) &&
           left.dflash_local_layer_bytes == right.dflash_local_layer_bytes &&
           left.image_bytes == right.image_bytes;
}

StateImageHostLayout plan_host_state_image(const StateImageSpec& spec) {
    if (spec.linear.layers == 0 || spec.linear.conv_channels <= 0 || spec.linear.conv_width <= 0 ||
        spec.linear.value_heads <= 0 || spec.linear.value_head_dim <= 0 ||
        spec.linear.key_head_dim <= 0 || spec.linear.slot_count <= 0 ||
        (spec.linear.conv_dtype != DType::BF16 && spec.linear.conv_dtype != DType::FP32) ||
        spec.hidden <= 0) {
        throw std::invalid_argument("StateImage host geometry is invalid");
    }
    if (spec.dflash_local && (spec.dflash_local->layers == 0 || spec.dflash_local->kv_heads <= 0 ||
                              spec.dflash_local->head_dim <= 0)) {
        throw std::invalid_argument("StateImage host DFlash geometry is invalid");
    }

    LayoutBuilder builder;
    StateImageHostLayout host;
    host.spec = spec;
    const Tensor conv_slot(nullptr, spec.linear.conv_dtype,
                           {spec.linear.conv_channels, spec.linear.conv_width});
    const Tensor hidden_slot(nullptr, DType::BF16, {spec.hidden});
    host.linear_conv_layer_bytes = conv_slot.bytes();
    host.linear_conv = builder.add(checked_mul(host.linear_conv_layer_bytes, spec.linear.layers,
                                               "StateImage host convolution bytes overflow"),
                                   kStateImageAlignment, "StateImage host convolution");
    host.linear_recurrent_layer_bytes =
        plan_recurrent_encoding(spec.linear, spec.host_recurrent, host.recurrent_encoding);
    host.linear_recurrent =
        builder.add(checked_mul(host.linear_recurrent_layer_bytes, spec.linear.layers,
                                "StateImage host recurrent bytes overflow"),
                    kStateImageAlignment, "StateImage host recurrent");
    host.continuation_hidden = builder.add(hidden_slot.bytes(), kStateImageAlignment,
                                           "StateImage host continuation hidden");
    if (spec.dflash_local) {
        const Tensor local_slot(
            nullptr, DType::BF16,
            {spec.dflash_local->head_dim,
             static_cast<std::int32_t>(padded_dflash_capacity(spec.dflash_local->capacity)),
             spec.dflash_local->kv_heads});
        host.dflash_local_layer_bytes = local_slot.bytes();
        const std::size_t component_bytes =
            checked_mul(host.dflash_local_layer_bytes, spec.dflash_local->layers,
                        "StateImage host DFlash local bytes overflow");
        host.dflash_local_k =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local K");
        host.dflash_local_v =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local V");
    }
    host.image_bytes = builder.finish(kStateImageAlignment, "StateImage host image");
    return host;
}

std::byte* byte_offset(std::byte* base, std::size_t offset) noexcept { return base + offset; }

const std::byte* byte_offset(const std::byte* base, std::size_t offset) noexcept {
    return base + offset;
}

void validate_slot(std::int32_t slot, std::int32_t slot_count, const char* label) {
    if (slot < 0 || slot >= slot_count) { throw std::out_of_range(label); }
}

} // namespace

StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                    const StateImageSpec& spec) {
    if (spec.hidden <= 0) {
        throw std::invalid_argument("StateImage hidden width must be positive");
    }

    StateImageDeviceLayout out;
    out.linear = plan_linear_attention_state_pool(builder, spec.linear);
    out.continuation_hidden =
        builder.add_tensor(DType::BF16, {spec.hidden, spec.linear.slot_count}, kStateImageAlignment,
                           "StateImage continuation hidden");
    if (spec.dflash_local) {
        const DFlashLocalStateSpec& dflash = *spec.dflash_local;
        out.dflash_local =
            plan_cyclic_kv_cache(builder, dflash.layers, dflash.capacity, dflash.kv_heads,
                                 dflash.head_dim, spec.linear.slot_count);
    }

    out.host = plan_host_state_image(spec);
    if (spec.host_recurrent != HostStateStorage::Fp32) {
        out.recurrent_staging = builder.add(out.host.linear_recurrent_layer_bytes,
                                            kStateImageAlignment, "StateImage recurrent staging");
    }
    return out;
}

TransferWork state_image_transfer_work(const StateImageHostLayout& layout) {
    std::size_t payload = checked_add(layout.linear_conv.bytes, layout.linear_recurrent.bytes,
                                      "StateImage transfer payload overflow");
    payload             = checked_add(payload, layout.continuation_hidden.bytes,
                                      "StateImage transfer payload overflow");
    if (layout.dflash_local_k) {
        if (!layout.spec.dflash_local || !layout.dflash_local_v) {
            throw std::invalid_argument("StateImage DFlash transfer layout is incomplete");
        }
        const std::size_t component_bytes =
            checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                        "StateImage transfer payload overflow");
        payload = checked_add(
            payload, checked_mul(component_bytes, 2U, "StateImage transfer payload overflow"),
            "StateImage transfer payload overflow");
    }
    // An encoded recurrent layer adds its encode or decode launch to the layer's copy.
    const std::uint64_t operations =
        2ULL * layout.spec.linear.layers + 1ULL +
        (layout.spec.host_recurrent != HostStateStorage::Fp32 ? layout.spec.linear.layers : 0ULL) +
        (layout.spec.dflash_local ? 2ULL * layout.spec.dflash_local->layers : 0ULL);
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout) {
    if (!layout.spec.dflash_local || !layout.dflash_local_k || !layout.dflash_local_v) {
        throw std::invalid_argument("StateImage has no DFlash local component");
    }
    const std::size_t component_bytes =
        checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                    "StateImage DFlash transfer payload overflow");
    const std::uint64_t operations = 2ULL * layout.spec.dflash_local->layers;
    if (component_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
        operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage DFlash transfer work exceeds its representation");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(2U * component_bytes),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

HostStatePool::HostStatePool(HostContextArena& arena, StateImageHostLayout layout)
    : layout_(std::move(layout)), arena_(&arena) {
    if (!same_host_layout(layout_, plan_host_state_image(layout_.spec))) {
        throw std::invalid_argument("HostStatePool image layout is invalid");
    }
    if (layout_.image_bytes < arena_->minimum_allocation_bytes()) {
        throw std::invalid_argument("Host state geometry is below the shared arena minimum");
    }
    const std::size_t maximum_slots = arena_->capacity_bytes() / layout_.image_bytes;
    if (maximum_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host state descriptor capacity exceeds uint32");
    }
    const auto capacity = static_cast<std::uint32_t>(maximum_slots);
    slots_.resize(capacity);
    free_slots_.resize(capacity);
    free_count_ = capacity;
    for (std::uint32_t index = 0; index < capacity; ++index) {
        free_slots_[index] = capacity - 1U - index;
    }
}

std::optional<HostStateSlotHandle> HostStatePool::allocate() noexcept {
    if (free_count_ == 0) { return std::nullopt; }
    auto storage = arena_->allocate(layout_.image_bytes, HostPlacement::High);
    if (!storage) { return std::nullopt; }
    const std::uint32_t index = free_slots_[--free_count_];
    Slot& slot                = slots_[index];
    slot.storage              = std::move(*storage);
    ++occupied_;
    return HostStateSlotHandle{.index = index, .generation = slot.generation, .owner = this};
}

bool HostStatePool::can_allocate() const noexcept {
    return free_count_ != 0 && arena_->can_allocate(layout_.image_bytes);
}

bool HostStatePool::publish(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    slots_[handle.index].storage.publish();
    return true;
}

bool HostStatePool::release(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    Slot& slot = slots_[handle.index];
    (void)slot.storage.release();
    if (++slot.generation == 0) { ++slot.generation; }
    free_slots_[free_count_++] = handle.index;
    --occupied_;
    return true;
}

HostStateImageView HostStatePool::writable_view(HostStateSlotHandle handle) {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

HostStateImageConstView HostStatePool::view(HostStateSlotHandle handle) const {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

std::uint32_t HostStatePool::capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
}

bool HostStatePool::valid(HostStateSlotHandle handle) const noexcept {
    return handle.owner == this && handle.index < slots_.size() &&
           slots_[handle.index].storage.valid() &&
           slots_[handle.index].generation == handle.generation;
}

std::byte* HostStatePool::slot_data(std::uint32_t index) const noexcept {
    return slots_[index].storage.data();
}

StateImageDevicePool::StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout)
    : linear_(backing, layout.linear),
      continuation_hidden_(layout.continuation_hidden.bind(backing)), host_layout_(layout.host) {
    if (continuation_hidden_.dtype != DType::BF16 || !continuation_hidden_.is_contiguous() ||
        continuation_hidden_.ne[0] != host_layout_.spec.hidden ||
        continuation_hidden_.ne[1] != linear_.slot_count()) {
        throw std::invalid_argument("StateImage continuation hidden layout is inconsistent");
    }
    if (layout.dflash_local.has_value() != host_layout_.spec.dflash_local.has_value()) {
        throw std::invalid_argument("StateImage DFlash layout is inconsistent");
    }
    StateImageSpec device_spec{.linear         = layout.linear.spec,
                               .hidden         = continuation_hidden_.ne[0],
                               .host_recurrent = host_layout_.spec.host_recurrent};
    if (layout.dflash_local) {
        device_spec.dflash_local = DFlashLocalStateSpec{
            .layers   = static_cast<std::uint32_t>(layout.dflash_local->k.size()),
            .capacity = layout.dflash_local->capacity,
            .kv_heads = layout.dflash_local->num_kv_heads,
            .head_dim = layout.dflash_local->head_dim,
        };
    }
    if (!same_host_layout(host_layout_, plan_host_state_image(device_spec))) {
        throw std::invalid_argument("StateImage host layout does not match its device components");
    }
    const bool encoded = host_layout_.spec.host_recurrent != HostStateStorage::Fp32;
    if (layout.recurrent_staging.has_value() != encoded ||
        (encoded && layout.recurrent_staging->bytes < host_layout_.linear_recurrent_layer_bytes)) {
        throw std::invalid_argument("StateImage recurrent staging does not match its encoding");
    }
    if (encoded) { recurrent_staging_ = layout.recurrent_staging->bind(backing); }
    if (layout.dflash_local) {
        if (layout.dflash_local->lane_capacity != linear_.slot_count()) {
            throw std::invalid_argument("StateImage components do not share one slot geometry");
        }
        dflash_local_.emplace(backing, *layout.dflash_local);
    }
}

StateImageDeviceSlotView StateImageDevicePool::slot_view(std::int32_t slot) const {
    StateImageDeviceSlotView view{
        .linear              = linear_.slot_view(slot),
        .continuation_hidden = continuation_hidden_slot(slot),
    };
    if (dflash_local_) { view.dflash_local = dflash_local_->slot_view(slot); }
    return view;
}

Tensor StateImageDevicePool::continuation_hidden_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage slot is out of range");
    return continuation_hidden_.slice(1, slot, 1).view({host_layout_.spec.hidden});
}

CyclicKVCache* StateImageDevicePool::dflash_local() noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

const CyclicKVCache* StateImageDevicePool::dflash_local() const noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

void StateImageDevicePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    validate_slot(slot, slot_count(), "StateImage zero slot is out of range");
    linear_.zero_slot(slot, stream);
    const Tensor hidden = continuation_hidden_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(hidden.data, 0, hidden.bytes(), stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, slot, 1);
            const Tensor v                    = view.v.slice(3, slot, 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::zero_all(cudaStream_t stream) {
    linear_.zero_all(stream);
    CUDA_CHECK(cudaMemsetAsync(continuation_hidden_.data, 0, continuation_hidden_.bytes(), stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::copy_slot(std::int32_t source, std::int32_t destination,
                                     cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage copy destination is out of range");
    if (source == destination) { return; }
    linear_.copy_slot(source, destination, stream);
    const Tensor source_hidden      = continuation_hidden_slot(source);
    const Tensor destination_hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(destination_hidden.data, source_hidden.data,
                               destination_hidden.bytes(), cudaMemcpyDeviceToDevice, stream));
    if (dflash_local_) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
    }
}

void StateImageDevicePool::copy_dflash_local(std::int32_t source, std::int32_t destination,
                                             cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage DFlash copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage DFlash copy destination is out of range");
    if (!dflash_local_) { throw std::logic_error("StateImage has no DFlash local component"); }
    if (source != destination) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
    }
}

void StateImageDevicePool::validate_host_layout(const StateImageHostLayout* layout,
                                                const std::byte* data) const {
    if (layout == nullptr || data == nullptr || !same_host_layout(*layout, host_layout_)) {
        throw std::invalid_argument("Host and device StateImage layouts do not match");
    }
}

void StateImageDevicePool::copy_to_host(std::int32_t source, HostStateImageView destination,
                                        cudaStream_t stream) const {
    validate_slot(source, slot_count(), "StateImage copy-to-host source is out of range");
    validate_host_layout(destination.layout, destination.data);
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        const Tensor conv = linear_.conv_slot(layer, source);
        CUDA_CHECK(cudaMemcpyAsync(
            byte_offset(destination.data, host_layout_.linear_conv.offset +
                                              layer * host_layout_.linear_conv_layer_bytes),
            conv.data, conv.bytes(), cudaMemcpyDeviceToHost, stream));
        const Tensor recurrent = linear_.recurrent_slot(layer, source);
        std::byte* host_layer =
            byte_offset(destination.data, host_layout_.linear_recurrent.offset +
                                              layer * host_layout_.linear_recurrent_layer_bytes);
        if (recurrent_staging_.data == nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(host_layer, recurrent.data, recurrent.bytes(),
                                       cudaMemcpyDeviceToHost, stream));
        } else {
            encode_recurrent_to_host(recurrent, host_layer, stream);
        }
    }
    const Tensor hidden = continuation_hidden_slot(source);
    CUDA_CHECK(
        cudaMemcpyAsync(byte_offset(destination.data, host_layout_.continuation_hidden.offset),
                        hidden.data, hidden.bytes(), cudaMemcpyDeviceToHost, stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, source, 1);
            const Tensor v                    = view.v.slice(3, source, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data, host_layout_.dflash_local_k->offset +
                                                  layer * host_layout_.dflash_local_layer_bytes),
                k.data, k.bytes(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data, host_layout_.dflash_local_v->offset +
                                                  layer * host_layout_.dflash_local_layer_bytes),
                v.data, v.bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }
}

void StateImageDevicePool::copy_from_host(HostStateImageConstView source, std::int32_t destination,
                                          cudaStream_t stream) {
    validate_slot(destination, slot_count(),
                  "StateImage copy-from-host destination is out of range");
    validate_host_layout(source.layout, source.data);
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        const Tensor conv = linear_.conv_slot(layer, destination);
        CUDA_CHECK(cudaMemcpyAsync(
            conv.data,
            byte_offset(source.data, host_layout_.linear_conv.offset +
                                         layer * host_layout_.linear_conv_layer_bytes),
            conv.bytes(), cudaMemcpyHostToDevice, stream));
        Tensor recurrent = linear_.recurrent_slot(layer, destination);
        const std::byte* host_layer =
            byte_offset(source.data, host_layout_.linear_recurrent.offset +
                                         layer * host_layout_.linear_recurrent_layer_bytes);
        if (recurrent_staging_.data == nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(recurrent.data, host_layer, recurrent.bytes(),
                                       cudaMemcpyHostToDevice, stream));
        } else {
            decode_recurrent_from_host(host_layer, recurrent, stream);
        }
    }
    const Tensor hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(hidden.data,
                               byte_offset(source.data, host_layout_.continuation_hidden.offset),
                               hidden.bytes(), cudaMemcpyHostToDevice, stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, destination, 1);
            const Tensor v                    = view.v.slice(3, destination, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                k.data,
                byte_offset(source.data, host_layout_.dflash_local_k->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                k.bytes(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                v.data,
                byte_offset(source.data, host_layout_.dflash_local_v->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                v.bytes(), cudaMemcpyHostToDevice, stream));
        }
    }
}

namespace {

struct RecurrentStagingViews {
    Tensor values;
    Tensor group_scales;
    Tensor slice_scales;
};

// Typed views of the staged encoded layer. The staging bytes mirror one Host recurrent layer, so
// a single contiguous copy moves the whole encoded layer.
RecurrentStagingViews recurrent_staging_views(const DeviceSpan& staging,
                                              const StateImageHostLayout& layout) {
    const LinearAttentionStatePoolSpec& linear = layout.spec.linear;
    const StateImageRecurrentEncoding& encoding = layout.recurrent_encoding;
    auto* base                                  = static_cast<std::byte*>(staging.data);
    const auto elements = static_cast<std::int32_t>(static_cast<std::int64_t>(linear.key_head_dim) *
                                                    linear.value_head_dim * linear.value_heads);
    RecurrentStagingViews views;
    if (layout.spec.host_recurrent == HostStateStorage::BFloat16) {
        views.values = Tensor(base, DType::BF16,
                              {linear.key_head_dim, linear.value_head_dim, linear.value_heads});
        return views;
    }
    views.values = Tensor(base, DType::U8, {elements / 2});
    views.group_scales =
        Tensor(base + encoding.group_scale_offset, DType::U8,
               {static_cast<std::int32_t>(elements / ops::kNvfp4SliceGroup)});
    views.slice_scales =
        Tensor(base + encoding.slice_scale_offset, DType::FP32, {linear.value_heads});
    return views;
}

} // namespace

void StateImageDevicePool::encode_recurrent_to_host(const Tensor& recurrent, std::byte* host_layer,
                                                    cudaStream_t stream) const {
    RecurrentStagingViews staged = recurrent_staging_views(recurrent_staging_, host_layout_);
    if (host_layout_.spec.host_recurrent == HostStateStorage::BFloat16) {
        ops::cast_fp32_to_bf16(recurrent, staged.values, stream);
    } else {
        ops::encode_nvfp4_slices(
            recurrent,
            static_cast<std::int64_t>(host_layout_.spec.linear.key_head_dim) *
                host_layout_.spec.linear.value_head_dim,
            staged.values, staged.group_scales, staged.slice_scales, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(host_layer, recurrent_staging_.data,
                               host_layout_.linear_recurrent_layer_bytes, cudaMemcpyDeviceToHost,
                               stream));
}

void StateImageDevicePool::decode_recurrent_from_host(const std::byte* host_layer,
                                                      Tensor& recurrent, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(recurrent_staging_.data, host_layer,
                               host_layout_.linear_recurrent_layer_bytes, cudaMemcpyHostToDevice,
                               stream));
    const RecurrentStagingViews staged = recurrent_staging_views(recurrent_staging_, host_layout_);
    if (host_layout_.spec.host_recurrent == HostStateStorage::BFloat16) {
        ops::cast_bf16_to_fp32(staged.values, recurrent, stream);
    } else {
        ops::decode_nvfp4_slices(
            staged.values, staged.group_scales, staged.slice_scales,
            static_cast<std::int64_t>(host_layout_.spec.linear.key_head_dim) *
                host_layout_.spec.linear.value_head_dim,
            recurrent, stream);
    }
}

} // namespace ninfer::models::qwen3_5
