#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/systemone.h"

#include "ninfer/types.h"

#include <functional>
#include <iostream>
#include <string>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// Returns the ApiError code a parse raises, or "" when it succeeds.
std::string rejection(const RequestJson& body) {
    try {
        (void)parse_systemone_request(body);
    } catch (const ApiException& exception) {
        return exception.error().code.empty() ? "rejected" : exception.error().code;
    }
    return "";
}

RequestJson rank_body(RequestJson criteria) {
    return RequestJson{
        {"model", "qwen3.8-27b"},
        {"state", "New request: kitchen faucet leaking"},
        {"questions",
         {{"similar",
           {{"type", "rank"},
            {"instructions", "Rate how similar each past work order is to the new request:"},
            {"criteria", std::move(criteria)}}}}},
    };
}

RequestJson numbered_criteria(std::size_t count) {
    RequestJson criteria = RequestJson::object();
    for (std::size_t i = 0; i < count; ++i) {
        criteria["WO-" + std::to_string(i)] = "candidate " + std::to_string(i);
    }
    return criteria;
}

} // namespace

int main() {
    int failures = 0;

    // Candidates are scored in key order, so the result cannot depend on the caller's order.
    const SystemOneRequest rank =
        parse_systemone_request(rank_body({{"b", "second"}, {"a", "first"}, {"c", "third"}}));
    failures += check(rank.questions.size() == 1, "rank request does not hold one question");
    const SystemOneQuestion& question = rank.questions.front();
    failures += check(question.kind == SystemOneKind::Rank, "rank question has the wrong kind");
    failures += check(question.keys == std::vector<std::string>{"a", "b", "c"},
                      "rank candidates are not sorted by key");
    failures += check(question.generation.score_branches ==
                          std::vector<std::string>{"0] ", "1] ", "2] "},
                      "rank branches do not name the sorted candidate ids");
    failures += check(question.generation.score_candidates ==
                          std::vector<std::string>{"0", "1", "2", "3", "4"},
                      "default rank scale does not score five bare digit levels");
    failures += check(question.threshold == 0.5, "rank threshold default is not 0.5");
    failures += check(!question.limit.has_value(), "rank limit is set by default");
    // Messages are system, user (state, candidates and scale), then the assistant prefill "[".
    const std::string& prompt = question.generation.messages.at(1).content.front().text;
    failures += check(prompt.find("[0] first") != std::string::npos &&
                          prompt.find("[2] third") != std::string::npos,
                      "rank prompt does not list the candidates under their ids");

    // Order invariance: the same candidates listed differently build the same request.
    const SystemOneRequest reordered =
        parse_systemone_request(rank_body({{"c", "third"}, {"a", "first"}, {"b", "second"}}));
    failures += check(
        reordered.questions.front().generation.messages.at(1).content.front().text == prompt,
        "rank prompt depends on the order the caller listed candidates");

    // A custom scale sets the level count; threshold and limit are carried through.
    RequestJson custom = rank_body({{"x", "only"}});
    custom["questions"]["similar"]["scale"]     = {"no", "partly", "yes"};
    custom["questions"]["similar"]["threshold"] = 0.7;
    custom["questions"]["similar"]["limit"]     = 10;
    const SystemOneQuestion custom_question = parse_systemone_request(custom).questions.front();
    failures += check(custom_question.generation.score_candidates.size() == 3,
                      "custom rank scale does not set the level count");
    failures += check(custom_question.threshold == 0.7 && custom_question.limit == 10U,
                      "rank threshold or limit is not carried through");

    // Up to the Engine's branch limit; three-digit ids still fit one branch.
    failures += check(rejection(rank_body(numbered_criteria(ninfer::kMaximumScoreBranches))).empty(),
                      "rank rejects the maximum candidate count");
    failures += check(rejection(rank_body(numbered_criteria(ninfer::kMaximumScoreBranches + 1))) ==
                          "too_many_candidates",
                      "rank accepts more candidates than the Engine scores");
    failures += check(rejection(rank_body(RequestJson::object())) == "too_many_candidates",
                      "rank accepts an empty candidate list");

    RequestJson bad_threshold = rank_body({{"x", "only"}});
    bad_threshold["questions"]["similar"]["threshold"] = 1.5;
    failures += check(!rejection(bad_threshold).empty(), "rank accepts a threshold above 1");
    RequestJson bad_scale = rank_body({{"x", "only"}});
    bad_scale["questions"]["similar"]["scale"] = {"only one level"};
    failures += check(!rejection(bad_scale).empty(), "rank accepts a one-level scale");

    // Images are base64, shared by every question and placed before the text of the user turn.
    RequestJson with_images = rank_body({{"x", "only"}});
    with_images["images"]   = {"iVBORw0KGgoAAAA=", "/9j/4AAQ"};
    with_images["questions"]["done"] = {{"type", "noul"}, {"instructions", "Is the task done?"}};
    const SystemOneRequest imaged    = parse_systemone_request(with_images);
    for (const SystemOneQuestion& question : imaged.questions) {
        const auto& content = question.generation.messages.at(1).content;
        failures += check(content.size() == 3 && content[0].kind == ContentKind::Image &&
                              content[1].kind == ContentKind::Image &&
                              content[2].kind == ContentKind::Text,
                          "images are not placed before the question text");
        failures += check(content[0].source.value == "data:image/png;base64,iVBORw0KGgoAAAA=" &&
                              content[1].source.media_type == "image/jpeg",
                          "image data is not carried as a base64 data source");
    }
    for (const char* source : {"data:image/png;base64,iVBORw0KGgo=", "https://x/a.png", ""}) {
        RequestJson bad_image = rank_body({{"x", "only"}});
        bad_image["images"]   = {source};
        failures += check(!rejection(bad_image).empty(), "systemone accepts a non-base64 image");
    }
    RequestJson object_images = rank_body({{"x", "only"}});
    object_images["images"]   = {{"data", "aGVsbG8="}};
    failures += check(!rejection(object_images).empty(), "systemone accepts a non-array images");

    // Instructions may be any nonempty JSON value; keep_alive is accepted and ignored.
    RequestJson json_instructions = rank_body({{"x", "only"}});
    json_instructions["questions"]["similar"]["instructions"] = {{"goal", "rate similarity"}};
    json_instructions["keep_alive"]                         = "5m";
    failures += check(rejection(json_instructions).empty(),
                      "systemone rejects JSON instructions or keep_alive");
    RequestJson empty_instructions = rank_body({{"x", "only"}});
    empty_instructions["questions"]["similar"]["instructions"] = RequestJson::object();
    failures += check(!rejection(empty_instructions).empty(),
                      "systemone accepts empty instructions");
    RequestJson bad_keep_alive = rank_body({{"x", "only"}});
    bad_keep_alive["keep_alive"] = RequestJson::array();
    failures += check(!rejection(bad_keep_alive).empty(), "systemone accepts an array keep_alive");

    // Every answer names its type, as the spec's answer objects do.
    {
        RequestJson typed = rank_body({{"x", "only"}});
        typed["questions"]["pick"] = {{"type", "choice"},
                                      {"instructions", "Which one?"},
                                      {"criteria", {{"a", nullptr}, {"b", "second"}}}};
        typed["questions"]["done"] = {{"type", "noul"}, {"instructions", "Done?"}};
        typed["questions"]["level"] = {
            {"type", "score"}, {"instructions", "How bad?"}, {"criteria", {"low", "high"}}};
        for (const SystemOneQuestion& question : parse_systemone_request(typed).questions) {
            GenerationOutcome outcome;
            const std::size_t values = question.kind == SystemOneKind::Rank
                                           ? question.keys.size() * question.legend.size()
                                           : question.generation.score_candidates.size();
            outcome.candidate_logprobs.assign(values, -1.0F);
            const RequestJson answer = make_systemone_answer(question, outcome);
            const char* expected     = question.name == "similar" ? "rank"
                                       : question.name == "pick"  ? "choice"
                                       : question.name == "done"  ? "noul"
                                                                  : "score";
            failures += check(answer.value("type", "") == expected, "answer type is missing");
        }
    }

    RequestJson empty_images = rank_body({{"x", "only"}});
    empty_images["images"] = RequestJson::array();
    failures += check(rejection(empty_images).empty(), "systemone rejects an empty images list");

    failures += check(!rejection(RequestJson{{"model", "m"}, {"state", "s"}}).empty(),
                      "systemone accepts a request without questions");
    RequestJson unknown = rank_body({{"x", "only"}});
    unknown["questions"]["similar"]["type"] = "essay";
    failures += check(!rejection(unknown).empty(), "systemone accepts an unknown question type");

    return failures == 0 ? 0 : 1;
}
