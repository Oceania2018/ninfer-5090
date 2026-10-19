#include "serve/systemone.h"

#include "ninfer/types.h"
#include "serve/generation_service.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <set>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::size_t kMaximumQuestions     = 64;
constexpr std::size_t kMaximumChoiceOptions = 26; // single-token labels A..Z
constexpr std::size_t kMaximumScoreLevels   = 10; // single-token labels 0..9
// A rank branch is "<id>] ", one token per digit plus "]" and " ": ids below 1000 take at most
// five tokens. The Engine caps the branch count.
constexpr std::size_t kMaximumRankCandidates = ninfer::kMaximumScoreBranches;
static_assert(ninfer::kMaximumScoreBranchTokens >= 5 && kMaximumRankCandidates <= 1000);

// Default rank scale, lowest first; callers replace it with "scale". Level 3 separates problems on
// the same item that may share the cause or the remedy (including vague complaints) from clearly
// different ones, which tracks graded human similarity judgments more closely than four levels.
const std::vector<std::string> kDefaultRankScale = {
    "unrelated item and problem",
    "different item with a similar symptom",
    "same item, clearly different problem",
    "same item, a problem that could share the cause or the remedy (including vague complaints "
    "such as 'not working')",
    // A symptom-only request matches a past order that diagnosed a cause of that symptom.
    "same underlying problem: the same item and the same failure, including a past order whose "
    "diagnosed cause explains the symptom in the new request",
};

constexpr std::string_view kDefaultSystem =
    "You answer decision questions about the given state. Follow the answer format exactly.";

[[noreturn]] void bad_request(std::string message, std::string param,
                              std::string code = "invalid_request") {
    throw ApiException(ApiError{.status  = 400,
                                .message = std::move(message),
                                .param   = std::move(param),
                                .code    = std::move(code)});
}

std::string require_text(const RequestJson& value, const std::string& param) {
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        bad_request(param + " must be a nonempty string", param);
    }
    return value.get<std::string>();
}

std::optional<std::string> optional_text(const RequestJson& object, const char* key,
                                         const std::string& param) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    return require_text(object.at(key), param);
}

const char* systemone_type_name(SystemOneKind kind) {
    switch (kind) {
    case SystemOneKind::Choice: return "choice";
    case SystemOneKind::Noul: return "noul";
    case SystemOneKind::Score: return "score";
    case SystemOneKind::Rank: return "rank";
    }
    return "choice";
}

// Spec: "a nonempty string or JSON value"; a JSON value is shown to the model as indented JSON.
std::string render_instructions(const RequestJson& value, const std::string& param) {
    if (value.is_string()) { return require_text(value, param); }
    if (value.is_null() || ((value.is_object() || value.is_array()) && value.empty())) {
        bad_request(param + " must be a nonempty string or JSON value", param);
    }
    return value.dump(2);
}

// Media type from the leading base64 characters, for logs only: decoding probes the bytes.
std::string_view sniff_image_type(std::string_view base64) {
    if (base64.starts_with("iVBORw0KGgo")) { return "image/png"; }
    if (base64.starts_with("/9j/")) { return "image/jpeg"; }
    if (base64.starts_with("UklGR")) { return "image/webp"; }
    if (base64.starts_with("R0lGOD")) { return "image/gif"; }
    if (base64.starts_with("Qk")) { return "image/bmp"; }
    return "application/octet-stream";
}

// Spec: base64-encoded images shared by every question, in request order; URLs and data URLs are
// not accepted. Decoding, size and vision checks happen with the rest of the request's media.
std::vector<ContentPart> parse_images(const RequestJson& body) {
    std::vector<ContentPart> parts;
    if (!body.contains("images") || body.at("images").is_null()) { return parts; }
    const RequestJson& images = body.at("images");
    if (!images.is_array()) {
        bad_request("images must be an array of base64-encoded images", "images");
    }
    for (std::size_t i = 0; i < images.size(); ++i) {
        const std::string param = "images[" + std::to_string(i) + "]";
        const std::string data  = require_text(images.at(i), param);
        if (data.starts_with("data:") || data.starts_with("http://") ||
            data.starts_with("https://")) {
            bad_request(param + " must be base64-encoded image data; URLs and data URLs are "
                                "not supported",
                        param);
        }
        ContentPart part;
        part.kind              = ContentKind::Image;
        part.type_raw          = "image";
        part.source.kind       = ninfer::product::media_acquire::SourceKind::Data;
        part.source.media_type = std::string(sniff_image_type(data));
        part.source.value      = "data:" + part.source.media_type + ";base64," + data;
        parts.push_back(std::move(part));
    }
    return parts;
}

std::string render_state(const RequestJson& body) {
    if (!body.contains("state")) { bad_request("missing required field: state", "state"); }
    const RequestJson& state = body.at("state");
    if (state.is_string()) { return require_text(state, "state"); }
    if ((state.is_object() || state.is_array()) && !state.empty()) { return state.dump(2); }
    bad_request("state must be a nonempty string, object or array", "state");
}

ChatTurn text_turn(ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    turn.content.push_back(ContentPart{.kind = ContentKind::Text, .text = std::move(text)});
    return turn;
}

// Scoring request: thinking off, one output position, and the final assistant prefill ends exactly
// where the scored token belongs.
GenerationRequest scoring_request(const std::string& system, std::string user,
                                  std::string assistant_prefill) {
    GenerationRequest request;
    request.messages.push_back(text_turn(ChatRole::System, system));
    request.messages.push_back(text_turn(ChatRole::User, std::move(user)));
    request.messages.push_back(text_turn(ChatRole::Assistant, std::move(assistant_prefill)));
    request.continuation              = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    request.enable_thinking           = false;
    request.chat_template_kwargs_json = R"({"enable_thinking":false})";
    request.max_tokens                = 1;
    request.sampling.temperature      = 0.0;
    return request;
}

std::string option_line(const std::string& label, const std::string& key,
                        const RequestJson& description) {
    std::string line = label + ". " + key;
    if (description.is_string() && !description.get_ref<const std::string&>().empty()) {
        line += ": " + description.get<std::string>();
    }
    return line;
}

void parse_choice(const RequestJson& item, const std::string& param, const std::string& state,
                  const std::string& instructions, const std::string& system,
                  SystemOneQuestion& question) {
    if (!item.contains("criteria") || !item.at("criteria").is_object()) {
        bad_request(param + ".criteria must be an object of options", param + ".criteria");
    }
    const RequestJson& criteria = item.at("criteria");
    if (criteria.size() < 2 || criteria.size() > kMaximumChoiceOptions) {
        bad_request(param + ".criteria must have 2 to " + std::to_string(kMaximumChoiceOptions) +
                        " options for this model",
                    param + ".criteria", "too_many_options");
    }
    std::string user = state + "\n\nQuestion: " + instructions + "\n\nOptions:";
    for (auto it = criteria.begin(); it != criteria.end(); ++it) {
        if (!it.value().is_null() && !it.value().is_string()) {
            bad_request(param + ".criteria values must be strings or null", param + ".criteria");
        }
        const std::string label(1, static_cast<char>('A' + question.keys.size()));
        user += "\n" + option_line(label, it.key(), it.value());
        question.generation.score_candidates.push_back(" " + label);
        question.keys.push_back(it.key());
    }
    user += "\n\nAnswer with the letter of the single best option.";
    GenerationRequest request = scoring_request(system, std::move(user), "Answer:");
    request.score_candidates  = std::move(question.generation.score_candidates);
    question.generation       = std::move(request);
}

void parse_noul(const RequestJson& item, const std::string& param, const std::string& state,
                const std::string& instructions, const std::string& system,
                SystemOneQuestion& question) {
    std::string user = state + "\n\nQuestion: " + instructions;
    if (item.contains("criteria") && !item.at("criteria").is_null()) {
        const RequestJson& criteria = item.at("criteria");
        if (!criteria.is_object()) {
            bad_request(param + ".criteria must be an object", param + ".criteria");
        }
        for (auto it = criteria.begin(); it != criteria.end(); ++it) {
            if (it.key() != "true" && it.key() != "false") {
                bad_request(param + ".criteria accepts only true and false", param + ".criteria");
            }
        }
        if (auto text = optional_text(criteria, "true", param + ".criteria.true")) {
            user += "\ntrue means: " + *text;
        }
        if (auto text = optional_text(criteria, "false", param + ".criteria.false")) {
            user += "\nfalse means: " + *text;
        }
    }
    user += "\n\nAnswer true or false.";
    question.generation                  = scoring_request(system, std::move(user), "Answer:");
    question.generation.score_candidates = {" true", " True", " false", " False"};
}

void parse_score(const RequestJson& item, const std::string& param, const std::string& state,
                 const std::string& instructions, const std::string& system,
                 SystemOneQuestion& question) {
    if (!item.contains("criteria") || !item.at("criteria").is_array()) {
        bad_request(param + ".criteria must be an array of level descriptions",
                    param + ".criteria");
    }
    const RequestJson& criteria = item.at("criteria");
    if (criteria.size() < 2 || criteria.size() > kMaximumScoreLevels) {
        bad_request(param + ".criteria must have 2 to 10 levels", param + ".criteria");
    }
    std::string user = state + "\n\nQuestion: " + instructions + "\n\nScale (lowest first):";
    std::vector<std::string> candidates;
    for (std::size_t level = 0; level < criteria.size(); ++level) {
        std::string description =
            require_text(criteria.at(level), param + ".criteria[" + std::to_string(level) + "]");
        user += "\n" + std::to_string(level) + ": " + description;
        question.legend.push_back(std::move(description));
        candidates.push_back(std::to_string(level));
    }
    // Qwen encodes " 7" as two tokens, so the prefill ends at "(" and the bare digit is scored.
    user += "\n\nAnswer with the number of the level that fits best, in parentheses.";
    question.generation                  = scoring_request(system, std::move(user), "Answer: (");
    question.generation.score_candidates = std::move(candidates);
}

// The candidate list, the rating scale and the reply format form one shared prefix that ends
// inside the assistant answer at "["; each candidate is then the branch "<id>] " and the rating
// digit that follows is scored.
void parse_rank(const RequestJson& item, const std::string& param, const std::string& state,
                const std::string& instructions, const std::string& system,
                SystemOneQuestion& question) {
    if (!item.contains("criteria") || !item.at("criteria").is_object()) {
        bad_request(param + ".criteria must be an object of keyed candidate texts",
                    param + ".criteria");
    }
    const RequestJson& criteria = item.at("criteria");
    if (criteria.empty() || criteria.size() > kMaximumRankCandidates) {
        bad_request(param + ".criteria must have 1 to " + std::to_string(kMaximumRankCandidates) +
                        " candidates",
                    param + ".criteria", "too_many_candidates");
    }
    std::vector<std::pair<std::string, std::string>> candidates;
    for (auto it = criteria.begin(); it != criteria.end(); ++it) {
        candidates.emplace_back(it.key(),
                                require_text(it.value(), param + ".criteria." + it.key()));
    }
    std::sort(candidates.begin(), candidates.end());
    if (item.contains("threshold") && !item.at("threshold").is_null()) {
        const RequestJson& threshold = item.at("threshold");
        if (!threshold.is_number() || threshold.get<double>() < 0.0 ||
            threshold.get<double>() > 1.0) {
            bad_request(param + ".threshold must be a number in [0, 1]", param + ".threshold");
        }
        question.threshold = threshold.get<double>();
    }
    if (item.contains("limit") && !item.at("limit").is_null()) {
        const RequestJson& limit = item.at("limit");
        if (!limit.is_number_integer() || limit.get<std::int64_t>() < 1) {
            bad_request(param + ".limit must be a positive integer", param + ".limit");
        }
        question.limit = static_cast<std::size_t>(limit.get<std::int64_t>());
    }
    if (item.contains("scale") && !item.at("scale").is_null()) {
        const RequestJson& scale = item.at("scale");
        if (!scale.is_array() || scale.size() < 2 || scale.size() > kMaximumScoreLevels) {
            bad_request(param + ".scale must be an array of 2 to 10 level descriptions",
                        param + ".scale");
        }
        for (std::size_t level = 0; level < scale.size(); ++level) {
            question.legend.push_back(require_text(
                scale.at(level), param + ".scale[" + std::to_string(level) + "]"));
        }
    } else {
        question.legend = kDefaultRankScale;
    }

    std::string user = state + "\n\nCandidates:";
    std::vector<std::string> branches;
    for (std::size_t id = 0; id < candidates.size(); ++id) {
        // Newlines inside a candidate would read as the start of the next candidate.
        std::string text = candidates[id].second;
        std::replace(text.begin(), text.end(), '\n', ' ');
        user += "\n[" + std::to_string(id) + "] " + text;
        question.keys.push_back(candidates[id].first);
        branches.push_back(std::to_string(id) + "] ");
    }
    user += "\n\n" + instructions + "\n";
    for (std::size_t level = question.legend.size(); level-- > 0;) {
        user += "\n" + std::to_string(level) + ": " + question.legend[level];
    }
    user += "\n\nI will name one candidate by its id. Reply with the id in brackets followed by "
            "its rating.";
    question.generation = scoring_request(system, std::move(user), "[");
    for (std::size_t level = 0; level < question.legend.size(); ++level) {
        question.generation.score_candidates.push_back(std::to_string(level));
    }
    question.generation.score_branches = std::move(branches);
}

std::vector<double> normalized(std::span<const float> logprobs) {
    double peak = -std::numeric_limits<double>::infinity();
    for (const float value : logprobs) { peak = std::max(peak, double(value)); }
    std::vector<double> out(logprobs.size(), 0.0);
    if (!std::isfinite(peak)) { return out; }
    double sum = 0.0;
    for (std::size_t i = 0; i < logprobs.size(); ++i) {
        out[i] = std::exp(double(logprobs[i]) - peak);
        sum += out[i];
    }
    for (double& value : out) { value /= sum; }
    return out;
}

// Spec: 1 - H(p)/ln(N); 0 for uniform probabilities, near 1 when one candidate dominates.
double confidence(const std::vector<double>& p) {
    double entropy = 0.0;
    for (const double value : p) {
        if (value > 0.0) { entropy -= value * std::log(value); }
    }
    return 1.0 - entropy / std::log(static_cast<double>(p.size()));
}

// Probability of the first half of `logprobs` (affirmative variants) against the second half.
double affirmative(std::span<const float> logprobs) {
    const std::vector<double> p = normalized(logprobs);
    const std::size_t half      = p.size() / 2;
    return std::accumulate(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(half), 0.0);
}

} // namespace

SystemOneRequest parse_systemone_request(const RequestJson& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object", ""); }
    SystemOneRequest out;
    if (!body.contains("model")) { bad_request("missing required field: model", "model"); }
    out.model = require_text(body.at("model"), "model");
    const std::vector<ContentPart> images = parse_images(body);
    // Accepted for compatibility; the model stays loaded for the life of the server.
    if (body.contains("keep_alive") && !body.at("keep_alive").is_null() &&
        !body.at("keep_alive").is_string() && !body.at("keep_alive").is_number()) {
        bad_request("keep_alive must be a duration string or a number of seconds", "keep_alive");
    }
    const std::string state  = render_state(body);
    const std::string system = optional_text(body, "system", "system")
                                   .value_or(std::string(kDefaultSystem));
    if (!body.contains("questions") || !body.at("questions").is_object() ||
        body.at("questions").empty() || body.at("questions").size() > kMaximumQuestions) {
        bad_request("questions must be an object of 1 to 64 named questions", "questions");
    }
    const RequestJson& questions = body.at("questions");
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        const std::string param = "questions." + it.key();
        const RequestJson& item = it.value();
        if (!item.is_object()) { bad_request(param + " must be an object", param); }
        if (!item.contains("type")) { bad_request(param + ".type is required", param + ".type"); }
        const std::string type = require_text(item.at("type"), param + ".type");
        if (!item.contains("instructions")) {
            bad_request(param + ".instructions is required", param + ".instructions");
        }
        const std::string instructions =
            render_instructions(item.at("instructions"), param + ".instructions");
        SystemOneQuestion question;
        question.name = it.key();
        if (type == "choice") {
            question.kind = SystemOneKind::Choice;
            parse_choice(item, param, state, instructions, system, question);
        } else if (type == "noul") {
            question.kind = SystemOneKind::Noul;
            parse_noul(item, param, state, instructions, system, question);
        } else if (type == "score") {
            question.kind = SystemOneKind::Score;
            parse_score(item, param, state, instructions, system, question);
        } else if (type == "rank") {
            question.kind = SystemOneKind::Rank;
            parse_rank(item, param, state, instructions, system, question);
        } else {
            bad_request(param + ".type must be choice, noul, score or rank", param + ".type");
        }
        // Images come before the text in the user turn, as in an ordinary vision prompt.
        if (!images.empty()) {
            std::vector<ContentPart>& content = question.generation.messages.at(1).content;
            content.insert(content.begin(), images.begin(), images.end());
        }
        out.questions.push_back(std::move(question));
    }
    return out;
}

RequestJson make_systemone_answer(const SystemOneQuestion& question,
                                  const GenerationOutcome& outcome) {
    const std::vector<float>& logprobs = outcome.candidate_logprobs;
    RequestJson answer                 = RequestJson::object();
    answer["type"]                     = systemone_type_name(question.kind);
    switch (question.kind) {
    case SystemOneKind::Choice: {
        const std::vector<double> p = normalized(logprobs);
        std::size_t best            = 0;
        RequestJson probabilities   = RequestJson::object();
        for (std::size_t i = 0; i < p.size(); ++i) {
            probabilities[question.keys[i]] = p[i];
            if (p[i] > p[best]) { best = i; } // ties keep the earlier option
        }
        answer["choice"]        = question.keys[best];
        answer["probabilities"] = std::move(probabilities);
        answer["confidence"]    = confidence(p);
        break;
    }
    case SystemOneKind::Noul: answer["noul"] = affirmative(logprobs); break;
    case SystemOneKind::Score: {
        const std::vector<double> p = normalized(logprobs);
        double score                = 0.0;
        RequestJson legend          = RequestJson::object();
        RequestJson probabilities   = RequestJson::object();
        for (std::size_t level = 0; level < p.size(); ++level) {
            score += static_cast<double>(level) * p[level];
            legend[std::to_string(level)]        = question.legend[level];
            probabilities[std::to_string(level)] = p[level];
        }
        answer["score"]         = score;
        answer["legend"]        = std::move(legend);
        answer["probabilities"] = std::move(probabilities);
        answer["confidence"]    = confidence(p);
        break;
    }
    case SystemOneKind::Rank: {
        const std::size_t levels = question.legend.size();
        if (logprobs.size() != question.keys.size() * levels) {
            throw std::logic_error("rank scoring returned an unexpected number of values");
        }
        // Similarity is the expected level over the rating distribution, scaled to [0, 1].
        std::vector<std::pair<double, std::size_t>> scored;
        for (std::size_t id = 0; id < question.keys.size(); ++id) {
            const std::vector<double> p =
                normalized(std::span<const float>(logprobs).subspan(id * levels, levels));
            double expected = 0.0;
            for (std::size_t level = 0; level < levels; ++level) {
                expected += static_cast<double>(level) * p[level];
            }
            scored.emplace_back(expected / static_cast<double>(levels - 1), id);
        }
        // Most similar first; equal similarities keep the sorted key order.
        std::stable_sort(scored.begin(), scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        RequestJson ranking      = RequestJson::array();
        RequestJson similarities = RequestJson::object();
        for (const auto& [similarity, id] : scored) {
            similarities[question.keys[id]] = similarity;
            if (similarity >= question.threshold &&
                (!question.limit || ranking.size() < *question.limit)) {
                ranking.push_back({{"key", question.keys[id]}, {"similarity", similarity}});
            }
        }
        RequestJson legend = RequestJson::object();
        for (std::size_t level = 0; level < levels; ++level) {
            legend[std::to_string(level)] = question.legend[level];
        }
        answer["ranking"]      = std::move(ranking);
        answer["similarities"] = std::move(similarities);
        answer["legend"]       = std::move(legend);
        answer["threshold"]    = question.threshold;
        break;
    }
    }
    return answer;
}

std::size_t systemone_output_tokens(const SystemOneQuestion& question) noexcept {
    return question.kind == SystemOneKind::Rank ? question.keys.size() : 1;
}

} // namespace ninfer::serve
