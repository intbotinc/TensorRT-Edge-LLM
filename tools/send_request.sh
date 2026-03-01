#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  send_request.sh --image <path> --text "<prompt>" --host <host> --port <port>

Notes:
  - Sends chat.completions with base64 image content.
  - image must be a local file; script embeds it as base64 bytes.
EOF
}

image_path=""
text_prompt=""
host=""
port=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      image_path="${2:-}"; shift 2;;
    --text)
      text_prompt="${2:-}"; shift 2;;
    --host)
      host="${2:-}"; shift 2;;
    --port)
      port="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ -z "$image_path" || -z "$text_prompt" || -z "$host" || -z "$port" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$image_path" ]]; then
  echo "Image not found: $image_path" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to build the JSON payload safely" >&2
  exit 1
fi

tmp_json="$(mktemp)"
python3 - "$image_path" "$text_prompt" "$tmp_json" <<'PY'
import base64
import json
import sys

image_path, text_prompt, out_path = sys.argv[1:4]
with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "model": "trt-edgellm",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image", "image": b64},
            ],
        }
    ],
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

curl -s "http://${host}:${port}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @"${tmp_json}"

rm -f "${tmp_json}"
