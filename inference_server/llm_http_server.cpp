/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 intbot inc. All rights reserved.
 */

#include "common/logger.h"
#include "common/trtUtils.h"
#include "http_server_utils.h"
#include "profileFormatter.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <getopt.h>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// cpp-httplib is expected at 3rdParty/httplib/httplib.h
#include "httplib.h"

using namespace trt_edgellm;
using Json = nlohmann::json;
namespace http = trt_edgellm::http_server;

namespace
{
struct EagleArgs
{
    bool enabled{false};
    int32_t draftTopK{10};
    int32_t draftStep{6};
    int32_t verifyTreeSize{60};
};

struct ServerArgs : public http::ServerArgs
{
    bool help{false};
    EagleArgs eagleArgs;
};

enum ServerOptionId : int
{
    HELP = 1000,
    ENGINE_DIR = 1001,
    MULTIMODAL_ENGINE_DIR = 1002,
    HOST = 1003,
    PORT = 1004,
    DEBUG = 1005,
    DEFAULT_MAX_GENERATE_LENGTH = 1006,
    DEFAULT_TEMPERATURE = 1007,
    DEFAULT_TOP_P = 1008,
    DEFAULT_TOP_K = 1009,
    MODEL_ID = 1010,
    EAGLE = 1011,
    EAGLE_DRAFT_TOP_K = 1012,
    EAGLE_DRAFT_STEP = 1013,
    EAGLE_VERIFY_TREE_SIZE = 1014,
    SAVE_IMAGE_TO_DISK = 1015,
    IMAGE_SAVE_DIR = 1016,
    MAX_IMAGE_BYTES = 1017,
    LOG_TO_CONSOLE = 1018,
    LOG_DIR = 1019,
    LOG_FILE_BASE_NAME = 1020,
    LOG_MAX_FILE_SIZE_MB = 1021,
    LOG_MAX_FILES = 1022,
    LOG_RETENTION_DAYS = 1023
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " --engineDir=<path> [--multimodalEngineDir=<path>] [--host=<host>] [--port=<port>]"
                 " [--debug] [--defaultMaxGenerateLength=<int>] [--defaultTemperature=<float>]"
                 " [--defaultTopP=<float>] [--defaultTopK=<int>] [--modelId=<string>] [--eagle]"
                 " [--eagleDraftTopK=<int>] [--eagleDraftStep=<int>] [--eagleVerifyTreeSize=<int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                        Display this help message" << std::endl;
    std::cerr << "  --engineDir                   Path to engine directory (required)" << std::endl;
    std::cerr << "  --multimodalEngineDir         Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --host                        Host to bind (default: 0.0.0.0)" << std::endl;
    std::cerr << "  --port                        Port to bind (default: 8080)" << std::endl;
    std::cerr << "  --debug                       Enable verbose logging" << std::endl;
    std::cerr << "  --defaultMaxGenerateLength    Default max tokens if request omits (default: 256)" << std::endl;
    std::cerr << "  --defaultTemperature          Default temperature (default: 1.0)" << std::endl;
    std::cerr << "  --defaultTopP                 Default top_p (default: 0.8)" << std::endl;
    std::cerr << "  --defaultTopK                 Default top_k (default: 50)" << std::endl;
    std::cerr << "  --modelId                     Model id string (default: trt-edgellm)" << std::endl;
    std::cerr << "  --eagle                       Enable Eagle speculative decoding mode" << std::endl;
    std::cerr << "  --eagleDraftTopK              Eagle draft top-k (default: 10)" << std::endl;
    std::cerr << "  --eagleDraftStep              Eagle draft steps (default: 6)" << std::endl;
    std::cerr << "  --eagleVerifyTreeSize         Eagle verify tree size (default: 60)" << std::endl;
    std::cerr << "  --saveImageToDisk             Save base64 images to disk (default: false)" << std::endl;
    std::cerr << "  --imageSaveDir                Image save directory (default: ./image_dumps)" << std::endl;
    std::cerr << "  --maxImageBytes               Maximum decoded image bytes (default: 10485760)" << std::endl;
    std::cerr << "  --logToConsole                Keep printing logs to stdout/stderr (default: false)" << std::endl;
    std::cerr << "  --logDir                      Log directory (default: ./logs)" << std::endl;
    std::cerr << "  --logFileBaseName             Log file base name (default: llm_http_server)" << std::endl;
    std::cerr << "  --logMaxFileSizeMB            Rotate size in MB (default: 100)" << std::endl;
    std::cerr << "  --logMaxFiles                 Max rotated files to keep (default: 20, 0 means unlimited)"
              << std::endl;
    std::cerr << "  --logRetentionDays            Delete files older than N days (default: 7, 0 means unlimited)"
              << std::endl;
}

bool parseServerArgs(ServerArgs& args, int argc, char* argv[])
{
    static struct option serverOptions[] = {{"help", no_argument, 0, ServerOptionId::HELP},
        {"engineDir", required_argument, 0, ServerOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, ServerOptionId::MULTIMODAL_ENGINE_DIR},
        {"host", required_argument, 0, ServerOptionId::HOST}, {"port", required_argument, 0, ServerOptionId::PORT},
        {"debug", no_argument, 0, ServerOptionId::DEBUG},
        {"defaultMaxGenerateLength", required_argument, 0, ServerOptionId::DEFAULT_MAX_GENERATE_LENGTH},
        {"defaultTemperature", required_argument, 0, ServerOptionId::DEFAULT_TEMPERATURE},
        {"defaultTopP", required_argument, 0, ServerOptionId::DEFAULT_TOP_P},
        {"defaultTopK", required_argument, 0, ServerOptionId::DEFAULT_TOP_K},
        {"modelId", required_argument, 0, ServerOptionId::MODEL_ID}, {"eagle", no_argument, 0, ServerOptionId::EAGLE},
        {"eagleDraftTopK", required_argument, 0, ServerOptionId::EAGLE_DRAFT_TOP_K},
        {"eagleDraftStep", required_argument, 0, ServerOptionId::EAGLE_DRAFT_STEP},
        {"eagleVerifyTreeSize", required_argument, 0, ServerOptionId::EAGLE_VERIFY_TREE_SIZE},
        {"saveImageToDisk", no_argument, 0, ServerOptionId::SAVE_IMAGE_TO_DISK},
        {"imageSaveDir", required_argument, 0, ServerOptionId::IMAGE_SAVE_DIR},
        {"maxImageBytes", required_argument, 0, ServerOptionId::MAX_IMAGE_BYTES},
        {"logToConsole", no_argument, 0, ServerOptionId::LOG_TO_CONSOLE},
        {"logDir", required_argument, 0, ServerOptionId::LOG_DIR},
        {"logFileBaseName", required_argument, 0, ServerOptionId::LOG_FILE_BASE_NAME},
        {"logMaxFileSizeMB", required_argument, 0, ServerOptionId::LOG_MAX_FILE_SIZE_MB},
        {"logMaxFiles", required_argument, 0, ServerOptionId::LOG_MAX_FILES},
        {"logRetentionDays", required_argument, 0, ServerOptionId::LOG_RETENTION_DAYS}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", serverOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case ServerOptionId::HELP: args.help = true; return true;
        case ServerOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case ServerOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case ServerOptionId::HOST: args.host = optarg; break;
        case ServerOptionId::PORT:
            args.port = std::stoi(optarg);
            if (args.port <= 0 || args.port > 65535)
            {
                LOG_ERROR("Invalid port: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::DEBUG: args.debug = true; break;
        case ServerOptionId::DEFAULT_MAX_GENERATE_LENGTH:
            args.defaultMaxGenerateLength = std::stoll(optarg);
            if (args.defaultMaxGenerateLength <= 0)
            {
                LOG_ERROR("Invalid defaultMaxGenerateLength: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::DEFAULT_TEMPERATURE: args.defaultTemperature = std::stof(optarg); break;
        case ServerOptionId::DEFAULT_TOP_P: args.defaultTopP = std::stof(optarg); break;
        case ServerOptionId::DEFAULT_TOP_K: args.defaultTopK = std::stoll(optarg); break;
        case ServerOptionId::MODEL_ID: args.modelId = optarg; break;
        case ServerOptionId::SAVE_IMAGE_TO_DISK: args.saveImageToDisk = true; break;
        case ServerOptionId::IMAGE_SAVE_DIR: args.imageSaveDir = optarg; break;
        case ServerOptionId::MAX_IMAGE_BYTES:
            args.maxImageBytes = static_cast<size_t>(std::stoull(optarg));
            if (args.maxImageBytes == 0)
            {
                LOG_ERROR("Invalid maxImageBytes: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::LOG_TO_CONSOLE: args.logToConsole = true; break;
        case ServerOptionId::LOG_DIR:
            args.logDir = optarg;
            if (args.logDir.empty())
            {
                LOG_ERROR("Invalid logDir: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::LOG_FILE_BASE_NAME:
            args.logFileBaseName = optarg;
            if (args.logFileBaseName.empty())
            {
                LOG_ERROR("Invalid logFileBaseName: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::LOG_MAX_FILE_SIZE_MB:
        {
            auto const maxFileSizeMb = std::stoull(optarg);
            constexpr uint64_t kMbToBytes = 1024ULL * 1024ULL;
            auto const maxSizeTByMb = static_cast<uint64_t>(std::numeric_limits<size_t>::max() / kMbToBytes);
            if (maxFileSizeMb == 0 || maxFileSizeMb > maxSizeTByMb)
            {
                LOG_ERROR("Invalid logMaxFileSizeMB: %s", optarg);
                return false;
            }
            args.logMaxFileBytes = static_cast<size_t>(maxFileSizeMb * kMbToBytes);
            break;
        }
        case ServerOptionId::LOG_MAX_FILES:
            args.logMaxFiles = std::stoi(optarg);
            if (args.logMaxFiles < 0)
            {
                LOG_ERROR("Invalid logMaxFiles: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::LOG_RETENTION_DAYS:
            args.logRetentionDays = std::stoi(optarg);
            if (args.logRetentionDays < 0)
            {
                LOG_ERROR("Invalid logRetentionDays: %s", optarg);
                return false;
            }
            break;
        case ServerOptionId::EAGLE: args.eagleArgs.enabled = true; break;
        case ServerOptionId::EAGLE_DRAFT_TOP_K: args.eagleArgs.draftTopK = std::stoi(optarg); break;
        case ServerOptionId::EAGLE_DRAFT_STEP: args.eagleArgs.draftStep = std::stoi(optarg); break;
        case ServerOptionId::EAGLE_VERIFY_TREE_SIZE: args.eagleArgs.verifyTreeSize = std::stoi(optarg); break;
        default: LOG_ERROR("Invalid Argument %c is %s.", opt, optarg); return false;
        }
    }

    if (args.engineDir.empty())
    {
        LOG_ERROR("--engineDir is required.");
        return false;
    }

    return true;
}

} // namespace

class InferenceService
{
public:
    explicit InferenceService(ServerArgs args)
        : mArgs(std::move(args))
    {
    }

    bool initialize()
    {
        auto pluginHandles = loadEdgellmPluginLib();
        (void) pluginHandles;

        CUDA_CHECK(cudaStreamCreate(&mStream));

        if (mArgs.debug)
        {
            gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
        }
        else
        {
            gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
        }

        try
        {
            if (mArgs.eagleArgs.enabled)
            {
                rt::EagleDraftingConfig draftingConfig{
                    mArgs.eagleArgs.draftTopK, mArgs.eagleArgs.draftStep, mArgs.eagleArgs.verifyTreeSize};
                mEagleRuntime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(mArgs.engineDir,
                    mArgs.multimodalEngineDir, std::unordered_map<std::string, std::string>{}, draftingConfig, mStream);
            }
            else
            {
                mRuntime = std::make_unique<rt::LLMInferenceRuntime>(mArgs.engineDir, mArgs.multimodalEngineDir,
                    std::unordered_map<std::string, std::string>{}, mStream);
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize runtime: %s", e.what());
            return false;
        }

        return true;
    }

    bool handleChatCompletion(Json const& body, Json& response, std::string& err, bool& isServerError)
    {
        auto const totalStart = std::chrono::steady_clock::now();
        isServerError = false;
        http::ParsedRequest parsed;
        if (!http::parseChatCompletionRequest(body, mArgs, parsed, err))
        {
            return false;
        }
        std::string clientSeqStr = parsed.clientSeq.has_value() ? std::to_string(*parsed.clientSeq) : "na";
        LOG_INFO("Request received: model=%s client_seq=%s", parsed.modelId.c_str(), clientSeqStr.c_str());
        /*//for debug only
                std::string textPrompt;
                for (auto const& singleRequest : parsed.request.requests)
                {
                    for (auto const& message : singleRequest.messages)
                    {
                        for (auto const& content : message.contents)
                        {
                            if (content.type != "text")
                            {
                                continue;
                            }
                            if (!textPrompt.empty())
                            {
                                textPrompt += "\n";
                            }
                            textPrompt += "[" + message.role + "] " + content.content;
                        }
                    }
                }
                if (textPrompt.empty())
                {
                    LOG_INFO("client_seq=%s, Text prompt: <empty>", clientSeqStr.c_str());
                }
                else
                {
                    LOG_INFO("client_seq=%s, Text prompt: %s", clientSeqStr.c_str(), textPrompt.c_str());
                }
        */
        rt::LLMGenerationResponse runtimeResponse;
        bool status = false;
        int64_t serverQueueWaitMs = 0;
        int64_t serverInferMs = 0;

        {
            auto const queueStart = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> guard(mMutex);
            auto const inferStart = std::chrono::steady_clock::now();
            serverQueueWaitMs = std::chrono::duration_cast<std::chrono::milliseconds>(inferStart - queueStart).count();
            if (mArgs.eagleArgs.enabled)
            {
                status = mEagleRuntime->handleRequest(parsed.request, runtimeResponse, mStream);
            }
            else
            {
                status = mRuntime->handleRequest(parsed.request, runtimeResponse, mStream);
            }
            auto const inferEnd = std::chrono::steady_clock::now();
            serverInferMs = std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd - inferStart).count();
        }
        auto const totalEnd = std::chrono::steady_clock::now();
        int64_t serverTotalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

        if (!status || runtimeResponse.outputTexts.empty())
        {
            LOG_ERROR("Request failed: client_seq=%s queue_wait_ms=%lld infer_ms=%lld total_ms=%lld",
                clientSeqStr.c_str(), static_cast<long long>(serverQueueWaitMs), static_cast<long long>(serverInferMs),
                static_cast<long long>(serverTotalMs));
            err = "Inference failed";
            isServerError = true;
            return false;
        }

        std::string outputText = sanitizeUtf8ForJson(runtimeResponse.outputTexts[0]);
        LOG_INFO("Request done: client_seq=%s queue_wait_ms=%lld infer_ms=%lld total_ms=%lld", clientSeqStr.c_str(),
            static_cast<long long>(serverQueueWaitMs), static_cast<long long>(serverInferMs),
            static_cast<long long>(serverTotalMs));
        LOG_INFO("client_seq=%s, Output text: %s", clientSeqStr.c_str(), outputText.c_str());
        Json choice;
        choice["index"] = 0;
        choice["message"]["role"] = "assistant";
        choice["message"]["content"] = outputText;
        choice["finish_reason"] = "stop";

        response["id"] = "cmpl-" + std::to_string(http::getCurrentUnixTimestamp());
        response["object"] = "chat.completion";
        response["created"] = http::getCurrentUnixTimestamp();
        response["model"] = parsed.modelId;
        response["client_seq"] = parsed.clientSeq.has_value() ? Json(*parsed.clientSeq) : Json(nullptr);
        response["choices"] = Json::array({choice});

        int64_t completionTokens = 0;
        if (!runtimeResponse.outputIds.empty())
        {
            completionTokens = static_cast<int64_t>(runtimeResponse.outputIds[0].size());
        }

        response["usage"]["prompt_tokens"] = 0;
        response["usage"]["completion_tokens"] = completionTokens;
        response["usage"]["total_tokens"] = completionTokens;
        response["metrics"]["server_queue_wait_ms"] = serverQueueWaitMs;
        response["metrics"]["server_infer_ms"] = serverInferMs;
        response["metrics"]["server_total_ms"] = serverTotalMs;

        return true;
    }

    Json getHealth() const
    {
        Json health;
        health["status"] = "ok";
        health["engine_dir"] = mArgs.engineDir;
        health["multimodal_engine_dir"] = mArgs.multimodalEngineDir;
        health["eagle"] = mArgs.eagleArgs.enabled;
        health["model_id"] = mArgs.modelId;
        return health;
    }

private:
    ServerArgs mArgs;
    cudaStream_t mStream{};
    std::unique_ptr<rt::LLMInferenceRuntime> mRuntime;
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> mEagleRuntime;
    std::mutex mMutex;
};

int main(int argc, char* argv[])
{
    ServerArgs args;
    if (!parseServerArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (!gLogger.setFileOutput(
            args.logDir, args.logFileBaseName, args.logMaxFileBytes, args.logMaxFiles, args.logRetentionDays))
    {
        std::cerr << "Failed to initialize log file at '" << args.logDir << "'" << std::endl;
        return EXIT_FAILURE;
    }
    gLogger.setConsoleEnabled(args.logToConsole);

    InferenceService service(args);
    if (!service.initialize())
    {
        return EXIT_FAILURE;
    }

    httplib::Server server;

    server.Get("/health", [&service](httplib::Request const&, httplib::Response& res) {
        Json body = service.getHealth();
        res.set_content(body.dump(2), "application/json");
    });

    server.Get("/v1/models", [&args](httplib::Request const&, httplib::Response& res) {
        Json body;
        body["object"] = "list";
        body["data"] = Json::array({{{"id", args.modelId}, {"object", "model"}}});
        res.set_content(body.dump(2), "application/json");
    });

    server.Post("/v1/chat/completions", [&service](httplib::Request const& req, httplib::Response& res) {
        Json body;
        try
        {
            body = Json::parse(req.body);
        }
        catch (Json::parse_error const& e)
        {
            Json err = http::makeErrorResponse("Invalid JSON payload");
            res.status = 400;
            res.set_content(err.dump(2), "application/json");
            return;
        }

        Json response;
        std::string errMsg;
        bool isServerError = false;
        try
        {
            if (!service.handleChatCompletion(body, response, errMsg, isServerError))
            {
                Json err = http::makeErrorResponse(errMsg, isServerError ? "server_error" : "invalid_request_error");
                res.status = isServerError ? 500 : 400;
                res.set_content(err.dump(2), "application/json");
                return;
            }
        }
        catch (std::exception const& e)
        {
            Json err = http::makeErrorResponse(std::string("Invalid request: ") + e.what(), "invalid_request_error");
            res.status = 400;
            res.set_content(err.dump(2), "application/json");
            return;
        }

        res.set_content(response.dump(2), "application/json");
    });

    LOG_INFO("Starting HTTP server on %s:%d", args.host.c_str(), args.port);
    server.listen(args.host, args.port);

    return EXIT_SUCCESS;
}
