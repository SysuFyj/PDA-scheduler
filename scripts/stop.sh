#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT"
STATE_DIR="${1:-$ROOT/outputs/latest}"

export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m vllm_pd_router.cluster stop --state-dir "$STATE_DIR"

