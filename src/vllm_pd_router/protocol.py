from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import urlparse


def build_prefill_payload(request_data: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(request_data)
    payload.pop("metadata", None)
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload.pop("max_completion_tokens", None)
    payload.pop("stream_options", None)
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_block_ids": None,
        "remote_engine_id": None,
        "remote_host": None,
        "remote_port": None,
    }
    return payload


def extract_kv_transfer_params(
    prefill_response: dict[str, Any],
    prefill_url: str,
    transfer_host: str | None = None,
) -> dict[str, Any]:
    raw = prefill_response.get("kv_transfer_params")
    if not isinstance(raw, dict):
        raise ValueError("Prefill response did not include kv_transfer_params")
    params = deepcopy(raw)
    host = transfer_host or urlparse(prefill_url).hostname
    if not host:
        raise ValueError(f"Cannot determine NIXL producer host from {prefill_url!r}")
    params["remote_host"] = host
    return params


def build_decode_payload(
    request_data: dict[str, Any], kv_transfer_params: dict[str, Any]
) -> dict[str, Any]:
    payload = deepcopy(request_data)
    payload.pop("metadata", None)
    payload["kv_transfer_params"] = deepcopy(kv_transfer_params)
    return payload


def forwarded_headers(headers: dict[str, str], request_id: str) -> dict[str, str]:
    result = {"x-request-id": request_id}
    for key in ("authorization", "x-custom-labels"):
        value = headers.get(key)
        if value:
            result[key] = value
    return result
