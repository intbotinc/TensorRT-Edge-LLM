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

#pragma once

#include "runtime/llmRuntimeUtils.h"

#include <atomic>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace http_server
{

struct ServerArgs
{
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string host{"0.0.0.0"};
    int32_t port{8080};
    bool debug{false};
    int64_t defaultMaxGenerateLength{256};
    float defaultTemperature{1.0f};
    float defaultTopP{0.8f};
    int64_t defaultTopK{50};
    std::string modelId{"trt-edgellm"};
    bool saveImageToDisk{false};
    std::string imageSaveDir{"./image_dumps"};
    size_t maxImageBytes{10 * 1024 * 1024};
    bool logToConsole{false};
    std::string logDir{"./logs"};
    std::string logFileBaseName{"llm_http_server"};
    size_t logMaxFileBytes{10 * 1024 * 1024};
    int32_t logMaxFiles{20};
    int32_t logRetentionDays{7};
};

struct ParsedRequest
{
    rt::LLMGenerationRequest request;
    std::string modelId;
    std::optional<int64_t> clientSeq;
};

inline nlohmann::json makeErrorResponse(std::string const& message, std::string const& type = "invalid_request_error")
{
    nlohmann::json err;
    err["error"]["message"] = message;
    err["error"]["type"] = type;
    return err;
}

inline int64_t getCurrentUnixTimestamp()
{
    return static_cast<int64_t>(std::time(nullptr));
}

inline size_t estimateDecodedSize(std::string const& input)
{
    if (input.size() < 4)
        return 0;
    size_t padding = 0;
    if (input.size() >= 2)
    {
        if (input[input.size() - 1] == '=')
            padding++;
        if (input[input.size() - 2] == '=')
            padding++;
    }
    return (input.size() / 4) * 3 - padding;
}

inline bool decodeBase64(std::string const& input, std::vector<unsigned char>& output, std::string& err)
{
    if (input.empty())
    {
        err = "base64 payload is empty";
        return false;
    }
    if (input.size() % 4 != 0)
    {
        err = "base64 length must be a multiple of 4";
        return false;
    }

    size_t padding = 0;
    if (input.size() >= 2)
    {
        if (input[input.size() - 1] == '=')
            padding++;
        if (input[input.size() - 2] == '=')
            padding++;
    }

    size_t decodedSize = (input.size() / 4) * 3 - padding;
    output.clear();
    output.reserve(decodedSize);

    for (size_t i = 0; i < input.size(); i += 4)
    {
        int8_t vals[4];
        for (int j = 0; j < 4; ++j)
        {
            unsigned char c = static_cast<unsigned char>(input[i + j]);
            if (c == '=')
            {
                if (i + j < input.size() - padding)
                {
                    err = "base64 has invalid padding";
                    return false;
                }
                vals[j] = 0;
                continue;
            }
            int8_t v = -1;
            if (c >= 'A' && c <= 'Z')
            {
                v = static_cast<int8_t>(c - 'A');
            }
            else if (c >= 'a' && c <= 'z')
            {
                v = static_cast<int8_t>(c - 'a' + 26);
            }
            else if (c >= '0' && c <= '9')
            {
                v = static_cast<int8_t>(c - '0' + 52);
            }
            else if (c == '+')
            {
                v = 62;
            }
            else if (c == '/')
            {
                v = 63;
            }
            if (v < 0)
            {
                err = "base64 contains invalid characters";
                return false;
            }
            vals[j] = v;
        }

        uint32_t triple = (static_cast<uint32_t>(vals[0]) << 18) | (static_cast<uint32_t>(vals[1]) << 12)
            | (static_cast<uint32_t>(vals[2] & 0x3F) << 6) | (static_cast<uint32_t>(vals[3] & 0x3F));

        output.push_back(static_cast<unsigned char>((triple >> 16) & 0xFF));
        if (input[i + 2] != '=')
            output.push_back(static_cast<unsigned char>((triple >> 8) & 0xFF));
        if (input[i + 3] != '=')
            output.push_back(static_cast<unsigned char>(triple & 0xFF));
    }

    if (output.size() != decodedSize)
    {
        err = "base64 decode size mismatch";
        return false;
    }
    return true;
}

inline std::string saveImageBytes(std::string const& dir, std::vector<unsigned char> const& bytes)
{
    static std::atomic<uint64_t> counter{0};
    std::filesystem::create_directories(dir);
    uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
    std::string filename = "image_" + std::to_string(getCurrentUnixTimestamp()) + "_" + std::to_string(id) + ".bin";
    std::filesystem::path path = std::filesystem::path(dir) / filename;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    return path.string();
}

inline bool parseChatCompletionRequest(
    nlohmann::json const& body, ServerArgs const& args, ParsedRequest& parsed, std::string& err)
{
    if (!body.contains("messages") || !body["messages"].is_array())
    {
        err = "'messages' array is required";
        return false;
    }
    if (body["messages"].empty())
    {
        err = "'messages' array must not be empty";
        return false;
    }

    rt::LLMGenerationRequest request;
    request.temperature = body.value("temperature", args.defaultTemperature);
    request.topP = body.value("top_p", args.defaultTopP);
    request.topK = body.value("top_k", args.defaultTopK);

    int64_t maxTokens = args.defaultMaxGenerateLength;
    if (body.contains("max_tokens"))
    {
        maxTokens = body.value("max_tokens", args.defaultMaxGenerateLength);
    }
    else if (body.contains("max_generate_length"))
    {
        maxTokens = body.value("max_generate_length", args.defaultMaxGenerateLength);
    }
    if (maxTokens <= 0)
    {
        err = "Invalid max_tokens or max_generate_length (must be positive)";
        return false;
    }
    request.maxGenerateLength = maxTokens;

    request.applyChatTemplate = body.value("apply_chat_template", true);
    request.addGenerationPrompt = body.value("add_generation_prompt", true);
    request.enableThinking = body.value("enable_thinking", false);
    // Default to false; client opts in with "save_system_prompt_kv_cache": true.
    // (Default was briefly true for TTFT savings, but combining it with engines
    // built with FP8 KV cache quantization caused CUDA illegal memory access on
    // the very first request. Safer to require opt-in.)
    request.saveSystemPromptKVCache = body.value("save_system_prompt_kv_cache", false);

    if (body.contains("lora_name") && !body["lora_name"].is_null())
    {
        err = "LoRA is not supported in this server build";
        return false;
    }

    rt::LLMGenerationRequest::Request single;
    auto const& messagesArray = body["messages"];

    for (auto const& messageJson : messagesArray)
    {
        if (!messageJson.contains("role") || !messageJson.contains("content"))
        {
            err = "Each message must have 'role' and 'content'";
            return false;
        }

        rt::Message msg;
        msg.role = messageJson["role"].get<std::string>();
        auto const& contentJson = messageJson["content"];

        if (contentJson.is_string())
        {
            rt::Message::MessageContent content;
            content.type = "text";
            content.content = contentJson.get<std::string>();
            msg.contents.push_back(std::move(content));
        }
        else if (contentJson.is_array())
        {
            for (auto const& contentItemJson : contentJson)
            {
                if (!contentItemJson.contains("type"))
                {
                    err = "Each content item must have a 'type' field";
                    return false;
                }

                rt::Message::MessageContent content;
                content.type = contentItemJson["type"].get<std::string>();

                if (content.type == "text")
                {
                    if (!contentItemJson.contains("text"))
                    {
                        err = "Content type 'text' requires 'text' field";
                        return false;
                    }
                    content.content = contentItemJson["text"].get<std::string>();
                }
                else if (content.type == "image")
                {
                    if (!contentItemJson.contains("image") || !contentItemJson["image"].is_string())
                    {
                        err = "Image content must include a base64 string 'image' field";
                        return false;
                    }
                    try
                    {
                        std::string b64 = contentItemJson["image"].get<std::string>();
                        size_t estimatedSize = estimateDecodedSize(b64);
                        if (estimatedSize > args.maxImageBytes)
                        {
                            err = "Decoded image exceeds maxImageBytes";
                            return false;
                        }
                        std::vector<unsigned char> bytes;
                        std::string decodeErr;
                        if (!decodeBase64(b64, bytes, decodeErr))
                        {
                            err = decodeErr;
                            return false;
                        }
                        if (bytes.size() > args.maxImageBytes)
                        {
                            err = "Decoded image exceeds maxImageBytes";
                            return false;
                        }
                        if (args.saveImageToDisk)
                        {
                            saveImageBytes(args.imageSaveDir, bytes);
                        }
                        auto image = rt::imageUtils::loadImageFromMemory(bytes.data(), bytes.size());
                        if (image.buffer == nullptr)
                        {
                            err = "Failed to decode image from base64";
                            return false;
                        }
                        content.content = "image";
                        single.imageBuffers.push_back(std::move(image));
                    }
                    catch (std::exception const& e)
                    {
                        err = e.what();
                        return false;
                    }
                }
                else
                {
                    err = "Unsupported content type: " + content.type;
                    return false;
                }

                msg.contents.push_back(std::move(content));
            }
        }
        else
        {
            err = "Message content must be a string or an array";
            return false;
        }

        single.messages.push_back(std::move(msg));
    }

    request.requests.push_back(std::move(single));
    parsed.request = std::move(request);
    parsed.modelId = body.value("model", args.modelId);
    parsed.clientSeq = std::nullopt;
    if (body.contains("client_seq"))
    {
        auto const& clientSeqJson = body["client_seq"];
        if (clientSeqJson.is_number_integer() || clientSeqJson.is_number_unsigned())
        {
            try
            {
                parsed.clientSeq = clientSeqJson.get<int64_t>();
            }
            catch (...)
            {
                // Keep compatibility: ignore malformed client_seq values.
                parsed.clientSeq = std::nullopt;
            }
        }
    }
    return true;
}

} // namespace http_server
} // namespace trt_edgellm
