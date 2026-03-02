/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference_server/http_server_utils.h"

#include <gtest/gtest.h>

using trt_edgellm::http_server::ParsedRequest;
using trt_edgellm::http_server::ServerArgs;

TEST(HttpServerUtils, MissingMessages)
{
    nlohmann::json body = nlohmann::json::object();
    ServerArgs args;
    ParsedRequest parsed;
    std::string err;

    bool ok = trt_edgellm::http_server::parseChatCompletionRequest(body, args, parsed, err);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(err.empty());
}

TEST(HttpServerUtils, EmptyMessages)
{
    nlohmann::json body;
    body["messages"] = nlohmann::json::array();

    ServerArgs args;
    ParsedRequest parsed;
    std::string err;

    bool ok = trt_edgellm::http_server::parseChatCompletionRequest(body, args, parsed, err);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(err.empty());
}

TEST(HttpServerUtils, ValidMinimalRequest)
{
    nlohmann::json body;
    body["messages"] = nlohmann::json::array({{{"role", "user"}, {"content", "Hello"}}});

    ServerArgs args;
    args.defaultTemperature = 0.5f;
    args.defaultTopP = 0.7f;
    args.defaultTopK = 33;
    args.defaultMaxGenerateLength = 77;
    args.modelId = "my-model";

    ParsedRequest parsed;
    std::string err;

    bool ok = trt_edgellm::http_server::parseChatCompletionRequest(body, args, parsed, err);
    EXPECT_TRUE(ok) << err;
    ASSERT_EQ(parsed.request.requests.size(), 1u);
    ASSERT_EQ(parsed.request.requests[0].messages.size(), 1u);
    ASSERT_EQ(parsed.request.requests[0].messages[0].contents.size(), 1u);
    EXPECT_EQ(parsed.request.requests[0].messages[0].role, "user");
    EXPECT_EQ(parsed.request.requests[0].messages[0].contents[0].type, "text");
    EXPECT_EQ(parsed.request.requests[0].messages[0].contents[0].content, "Hello");

    EXPECT_FLOAT_EQ(parsed.request.temperature, 0.5f);
    EXPECT_FLOAT_EQ(parsed.request.topP, 0.7f);
    EXPECT_EQ(parsed.request.topK, 33);
    EXPECT_EQ(parsed.request.maxGenerateLength, 77);
    EXPECT_TRUE(parsed.request.applyChatTemplate);
    EXPECT_TRUE(parsed.request.addGenerationPrompt);
    EXPECT_FALSE(parsed.request.enableThinking);
    EXPECT_EQ(parsed.modelId, "my-model");
}

TEST(HttpServerUtils, RejectsLora)
{
    nlohmann::json body;
    body["messages"] = nlohmann::json::array({{{"role", "user"}, {"content", "Hello"}}});
    body["lora_name"] = "adapter";

    ServerArgs args;
    ParsedRequest parsed;
    std::string err;

    bool ok = trt_edgellm::http_server::parseChatCompletionRequest(body, args, parsed, err);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(err.empty());
}

TEST(HttpServerUtils, ValidBase64ImageRequest)
{
    // 1x1 RGB BMP (red pixel)
    std::string const bmpBase64 = "Qk06AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABABgAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AA==";

    nlohmann::json body;
    body["messages"] = nlohmann::json::array({
        {{"role", "user"},
            {"content", nlohmann::json::array({{{"type", "text"}, {"text", "describe"}},
                {{"type", "image"}, {"image", bmpBase64}}})}}});

    ServerArgs args;
    args.maxImageBytes = 10 * 1024 * 1024;

    ParsedRequest parsed;
    std::string err;

    bool ok = trt_edgellm::http_server::parseChatCompletionRequest(body, args, parsed, err);
    EXPECT_TRUE(ok) << err;
    ASSERT_EQ(parsed.request.requests.size(), 1u);
    EXPECT_EQ(parsed.request.requests[0].imageBuffers.size(), 1u);
}

TEST(HttpServerUtils, RejectsInvalidBase64)
{
    nlohmann::json body;
    body["messages"] = nlohmann::json::array({
        {{"role", "user"},
            {"content", nlohmann::json::array({{{"type", "image"}, {"image", "not_base64"}}})}}});

    ServerArgs args;
    ParsedRequest parsed;
    std::string err;

    bool ok = trt_edgellm::http_server::parseChatCompletionRequest(body, args, parsed, err);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(err.empty());
}

TEST(HttpServerUtils, RejectsOversizedBase64)
{
    std::string const bmpBase64 = "Qk06AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABABgAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AA==";

    nlohmann::json body;
    body["messages"] = nlohmann::json::array({
        {{"role", "user"},
            {"content", nlohmann::json::array({{{"type", "image"}, {"image", bmpBase64}}})}}});

    ServerArgs args;
    args.maxImageBytes = 1;

    ParsedRequest parsed;
    std::string err;

    bool ok = trt_edgellm::http_server::parseChatCompletionRequest(body, args, parsed, err);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(err.empty());
}
