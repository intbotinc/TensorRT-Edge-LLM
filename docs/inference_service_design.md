# TensorRT-Edge-LLM C++ HTTP Inference Service Design (No Streaming)

## Goal
Build a single-process C++ HTTP service that exposes an OpenAI-compatible API (non-streaming) using the existing runtime flow from `examples/llm/llm_inference.cpp`. The server keeps TensorRT engines loaded and handles requests with low latency.

## Non-Goals
- Streaming responses
- Multi-process architecture
- Python/FastAPI dependency

## Summary
- Single C++ binary
- HTTP server based on `cpp-httplib`
- OpenAI-compatible JSON API
- Reuse `LLMInferenceRuntime` / `LLMInferenceSpecDecodeRuntime`
- Simple concurrency (one request at a time, mutex protected)

---

## 1. API Surface

### Endpoints
- `GET /health`
  - Returns service status and engine readiness
- `GET /v1/models`
  - Returns available model(s)
- `POST /v1/chat/completions`
  - Main inference endpoint (OpenAI-compatible, non-streaming)
- (Optional) `POST /v1/completions`
  - Map to a single-prompt chat completion

### Request (Chat Completions)
```json
{
  "model": "trt-edgellm",
  "client_seq": 4227,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Hello"}
      ]
    }
  ],
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 256
}
```

`client_seq` is optional. The server must accept requests with or without this field.

### Response
```json
{
  "id": "cmpl-123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "trt-edgellm",
  "client_seq": 4227,
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hi!"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 3,
    "total_tokens": 15
  },
  "metrics": {
    "server_total_ms": 805,
    "server_infer_ms": 790
  }
}
```

---

## 2. Core Components

### A. HTTP Server Layer
Use `cpp-httplib` for a small, dependency-light server:
- Header-only, easy to integrate
- Supports JSON body parsing

### B. Runtime Manager
Owns and manages:
- `LLMInferenceRuntime` (standard)
- `LLMInferenceSpecDecodeRuntime` (Eagle, optional)
- `cudaStream_t`

Responsibilities:
- Engine initialization (startup)
- Request execution
- Error propagation

### C. Request Translator
Maps OpenAI-compatible JSON -> internal request:
- `messages[]` -> `rt::Message`
- `content` array -> `MessageContent`
- Sampling params -> internal config

### D. Response Builder
Maps internal response -> OpenAI-compatible JSON
- Echo optional `client_seq` when provided by client
- Include per-request timing metrics in response for tracing

---

## 3. Concurrency Model

**Baseline:** single request at a time
- Mutex around runtime invocation
- Predictable GPU usage

**Optional upgrade:** queue + worker thread
- Enqueue request, wait for completion

---

## 4. Input Mapping Details

### Messages
- `role` -> `rt::Message.role`
- `content` items:
  - Text: `{ "type": "text", "text": "..." }`
  - Image: `{ "type": "image", "image": "<BASE64_BYTES>" }`

### Multimodal Images
Only this format is supported:

```json
{
  "type": "image",
  "image": "<BASE64_BYTES>"
}
```

Constraints and validation:
- Base64 must be valid.
- The decoded image size must be <= `maxImageBytes` (default 10 MB, configurable).

Processing behavior:
- Decode from memory (no disk writes in normal mode).
- If debug saving is enabled, images are saved to disk for inspection.

---

## 5. Error Handling

Return JSON errors with HTTP status:
- 400: invalid request
- 500: inference/runtime error
- 503: engine not ready

Error format:
```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error"
  }
}
```

---

## 6. Configuration (CLI)

Proposed server CLI options:
- `--engineDir <path>`
- `--multimodalEngineDir <path>`
- `--port <int>`
- `--debug`
- `--defaultMaxGenerateLength <int>`
- `--defaultTemperature <float>`
- `--defaultTopP <float>`
- `--defaultTopK <int>`
- `--modelId <string>`
- `--eagle`
- `--eagleDraftTopK <int>`
- `--eagleDraftStep <int>`
- `--eagleVerifyTreeSize <int>`

Image-related:
- `--saveImageToDisk <bool>` (default: false, debug only)
- `--imageSaveDir <string>` (default: `./image_dumps`)
- `--maxImageBytes <int>` (default: `10485760`)

---

## 7. Startup Flow

1. Parse CLI args
2. Initialize CUDA stream
3. Load engines
4. Start HTTP server

---

## 8. Request Flow

1. HTTP request arrives
2. Parse JSON body
3. Extract optional `client_seq` (if missing, continue with `client_seq = null`)
4. Validate required fields
5. Build `rt::LLMGenerationRequest`
6. Run `handleRequest()`
7. Measure and log per-request timing (`server_total_ms`, `server_infer_ms`) with `client_seq`
8. Build JSON response (including optional `client_seq` and timing metrics)
9. Return HTTP response

---

## 9. Request Correlation and Compatibility

- Correlation key is `client_seq` from request JSON body.
- No HTTP header is required for request correlation.
- If `client_seq` is missing or invalid type, the server must still process the request.
- Server logs should print one line per request with `client_seq` and elapsed timing.
- Client and server logs can be joined by `client_seq` for latency analysis.

---

## 10. Future Extensions

- Add streaming
- Add batching + queue
- Add LoRA per-request selection
- Add remote URL fetch with security controls

---

## 11. File Layout Proposal

- `examples/llm/llm_http_server.cpp` (new main)
- `examples/llm/http/` (optional helpers)
- `docs/inference_service_design.md` (this doc)

---

## 12. Why This Design

- Minimal changes to existing runtime flow
- Keeps GPU engine warm for low latency
- Simple deployment: one binary
- Compatible with OpenAI-style clients (non-streaming)
