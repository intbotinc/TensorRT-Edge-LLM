# LLM HTTP Server Usage

This document describes how to build and run the `llm_http_server` example and how to call its API using **OpenAI-compatible** request formats.

## Build

The HTTP server depends on cpp-httplib. Ensure the header is present:

- `3rdParty/httplib/httplib.h`

Configure and build:

```bash
cmake -S . -B build
cmake --build build --target llm_http_server
```

If you also want unit tests:

```bash
cmake -S . -B build -DBUILD_UNIT_TESTS=ON
cmake --build build --target llm_http_server unitTest
```

## Run

```bash
./build/examples/llm/llm_http_server \
  --engineDir /path/to/text/engine \
  --multimodalEngineDir /path/to/visual/engine \
  --host 0.0.0.0 \
  --port 8080
```

Notes:
- `--multimodalEngineDir` is optional.
- For Eagle mode, add `--eagle` and its parameters.

### Example: Eagle mode

```bash
./build/examples/llm/llm_http_server \
  --engineDir /path/to/text/engine \
  --multimodalEngineDir /path/to/visual/engine \
  --eagle \
  --eagleDraftTopK 10 \
  --eagleDraftStep 6 \
  --eagleVerifyTreeSize 60
```

## API

### Health

```bash
curl http://localhost:8080/health
```

### Model List

```bash
curl http://localhost:8080/v1/models
```

### Chat Completions (Text)

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "trt-edgellm",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Say hello in one sentence."}
        ]
      }
    ],
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 128
  }'
```

### Chat Completions (JSON + Image)

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "trt-edgellm",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Here is metadata: {\"task\":\"classify\",\"priority\":2}"},
          {
            "type": "image",
            "image": "<BASE64_BYTES>"
          }
        ]
      }
    ]
  }'
```

## Request Fields

- `messages` (required): Array of messages with `role` and `content`.
- `content` items:
  - Text: `{ "type": "text", "text": "..." }`
- Image: `{ "type": "image", "image": "<BASE64_BYTES>" }`
- Sampling fields (optional): `temperature`, `top_p`, `top_k`, `max_tokens`
- Optional toggles: `apply_chat_template`, `add_generation_prompt`, `enable_thinking`

## Image Rules and Limits

- Only base64 image format is supported (`type: image`).
- Default max decoded image size: 10 MB (configurable with `--maxImageBytes`).

## Debug Image Save

- `--saveImageToDisk` (default: false)
- `--imageSaveDir` (default: `./image_dumps`)

When enabled, base64 images are saved to disk for debugging.

## Response Shape

```json
{
  "id": "cmpl-123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "trt-edgellm",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 42,
    "total_tokens": 42
  }
}
```

## Notes and Limitations

- Non-streaming responses only.
- Single request at a time (mutex-protected runtime).
- LoRA selection is not supported in this server build.
- Image inputs must be provided as base64 in `{ "type": "image", "image": "<BASE64_BYTES>" }`.
