#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/systemone.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

// The System One spec's error body is a single string: {"error": "..."}.
void write_systemone_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(RequestJson{{"error", error.message}}.dump(), "application/json");
}

} // namespace

// Questions run one after another as ordinary Engine requests, so each is scheduled, logged and
// cancelled like any other request; answers are independent and appear under their names.
void HttpServer::handle_systemone(const httplib::Request& req, httplib::Response& res) {
    SystemOneRequest request;
    try {
        request = parse_systemone_request(parse_json_body(req));
        validate_openai_model(request.model, public_model_id_);
    } catch (const ApiException& exception) {
        write_systemone_error(res, exception.error());
        return;
    }

    const auto internal_error = [](const char* message) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = message;
        return error;
    };
    RequestJson answers             = RequestJson::object();
    std::uint64_t input_tokens      = 0;
    std::uint64_t cached_tokens     = 0;
    std::uint64_t output_tokens     = 0;
    for (const SystemOneQuestion& question : request.questions) {
        const std::uint64_t req_id = ++request_seq_;
        RequestLogMetadata metadata{.http_request_id        = res.get_header_value("x-request-id"),
                                    .model                  = request.model,
                                    .stream                 = false,
                                    .output_tokens_explicit = true};
        PreparedRequest prepared;
        try {
            ninfer::GenerationObservationOptions observation{
                .phase_timings = true,
                .scheduling    = scheduling_observer(req_id, metadata.http_request_id),
                .first_token   = first_token_observer(),
            };
            prepared = service_->prepare(question.generation, GenerationConsumerMode::Aggregate,
                                         std::move(observation),
                                         [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            record_request_rejected(make_request_rejection_log_context(
                req_id, "systemone", question.generation, metadata, exception.error()));
            write_systemone_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const ApiError error = internal_error(exception.what());
            record_request_rejected(make_request_rejection_log_context(
                req_id, "systemone", question.generation, metadata, error));
            write_systemone_error(res, error);
            return;
        }

        auto lifecycle = begin_request(
            make_request_log_context(req_id, "systemone", question.generation, metadata, prepared));
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_systemone_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            write_systemone_error(res, internal_error(exception.what()));
            return;
        }
        lifecycle->done(outcome);
        try {
            answers[question.name] = make_systemone_answer(question, outcome);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            write_systemone_error(res, internal_error(exception.what()));
            return;
        }
        input_tokens += static_cast<std::uint64_t>(outcome.prompt_tokens);
        cached_tokens += std::min<std::uint64_t>(outcome.metrics.prefix_cache_hit_tokens,
                                                 static_cast<std::uint64_t>(outcome.prompt_tokens));
        output_tokens += systemone_output_tokens(question);
    }

    RequestJson payload = RequestJson::object();
    payload["model"]    = request.model;
    payload["answers"]  = std::move(answers);
    // cached_input_tokens is the part of input_tokens served from the prefix cache.
    payload["usage"]    = {{"input_tokens", input_tokens},
                           {"cached_input_tokens", cached_tokens},
                           {"output_tokens", output_tokens}};
    res.set_content(payload.dump(), "application/json");
}

} // namespace ninfer::serve
