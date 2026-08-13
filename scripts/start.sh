#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT"
CONFIG="${1:-$PROJECT/configs/2p4d.yaml}"
STATE_DIR="${2:-$ROOT/outputs/latest}"

export VLLM_PD_MODEL="${VLLM_PD_MODEL:?Set VLLM_PD_MODEL to the local model path}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m vllm_pd_router.cluster start --config "$CONFIG" --state-dir "$STATE_DIR"

