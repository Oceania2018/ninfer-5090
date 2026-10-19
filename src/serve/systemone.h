#pragma once

// Ollama-compatible System One decision endpoint (POST /v1/systemone). Every question becomes one
// scoring GenerationRequest whose answer is computed from candidate-token log probabilities at a
// fixed position; nothing is decoded. Besides the standard choice, noul and score questions, the
// NInfer extension "rank" grades up to 256 keyed candidates against the shared state in a single
// request (one prefill plus one short branch per candidate) on an ordered similarity scale, and
// returns every candidate's expected similarity in [0, 1], sorted, with the ones at or above a
// threshold listed as the ranking. As in the spec (docs.ollama.com/api/systemone), "images" are
// base64 strings shared by every question, answers carry their "type", and errors are
// {"error": "<message>"}. Extensions: "rank", "system", and usage.cached_input_tokens.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

enum class SystemOneKind : std::uint8_t { Choice, Noul, Score, Rank };

struct SystemOneQuestion {
    std::string name;
    SystemOneKind kind = SystemOneKind::Choice;
    // choice: option keys in request order. rank: candidate keys in scoring order (sorted, so the
    // result does not depend on the order the caller listed them).
    std::vector<std::string> keys;
    std::vector<std::string> legend; // score, rank: level descriptions, lowest first
    double threshold = 0.5;          // rank: minimum similarity listed in the ranking
    std::optional<std::size_t> limit; // rank: maximum ranking length
    GenerationRequest generation;
};

struct SystemOneRequest {
    std::string model;
    std::vector<SystemOneQuestion> questions;
};

SystemOneRequest parse_systemone_request(const RequestJson& body);

// One answer object for the response's "answers" map.
RequestJson make_systemone_answer(const SystemOneQuestion& question,
                                  const GenerationOutcome& outcome);

// Internal scoring positions a question consumes ("usage.output_tokens").
std::size_t systemone_output_tokens(const SystemOneQuestion& question) noexcept;

} // namespace ninfer::serve
