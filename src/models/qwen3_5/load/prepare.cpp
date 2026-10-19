#include "models/qwen3_5/load/bindings.h"

#include "artifact/views.h"

#include <cmath>
#include <variant>

namespace ninfer::models::qwen3_5::loading {

#if defined(NINFER_SM86)
namespace {

// True when every object backing the parameter is a format whose A8 route is the int8
// (q8_act + mma.s8) path rather than FP8/TMA.
bool sm89_int8_activation_format(artifact::Binder& binder,
                                 const artifact::ParameterReference& reference) {
    if (reference.binding.parts.empty()) { return false; }
    for (const auto& part : reference.binding.parts) {
        const auto& object = binder.reader().directory().object(part.object);
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr || (tensor->format != "iq3_xxs" && tensor->format != "q4_g64_fp16" &&
                                  tensor->format != "q5_g64_fp16")) {
            return false;
        }
    }
    return true;
}

} // namespace
#endif

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference =
        binder.parameter(name, std::move(shape), artifact::Residency::Device, exact_format);
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
#if defined(NINFER_SM86)
        // sm_89 build: the FP8 A8 and NVFP4 A4 routes load through TMA or use E2M1
        // block-scale MMA, which sm_89 lacks, so those weights run their A16 routes. The IQ3_XXS
        // and Q4/Q5 group-64 A8 routes quantize activations to int8 and contract with
        // mma.m16n8k32.s8 over cp.async (sm_80+), so they run as AllowA8 for prefill. AllowA4
        // uses get the same int8 route: artifacts that copy the NVFP4 package's policies tag the
        // dense MLP AllowA4, and A4 is a superset of A8.
        case artifact::ActivationPolicy::AllowA8:
        case artifact::ActivationPolicy::AllowA4:
            result.policy = sm89_int8_activation_format(binder, pending.reference)
                                ? ops::LinearPolicy::AllowA8
                                : ops::LinearPolicy::A16Only;
            break;
#else
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
#endif
        }
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return parameter(std::move(name), std::move(shape), {}, format);
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
