#!/usr/bin/env bash
set -euo pipefail

ROUTER_URL="${1:-http://127.0.0.1:8300}"
MODEL="${2:-${VLLM_PD_MODEL:?Set VLLM_PD_MODEL or pass a model name explicitly}}"

curl -N -sS "$ROUTER_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<JSON
{
  "model": "$MODEL",
  "prompt": "The capital of France is",
  "max_tokens": 16,
  "temperature": 0,
  "stream": true
}
JSON
)"

