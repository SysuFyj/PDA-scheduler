from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"Trace is empty: {path}")
    return rows


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def requested_output_tokens(row: dict[str, Any], cap: int) -> int:
    value = (
        row.get("target_output_tokens")
        or row.get("oracle_output_tokens")
        or row.get("reference_output_tokens")
        or row.get("qwen_generated_tokens")
        or cap
    )
    return min(max(1, int(value)), cap)


async def run_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    row: dict[str, Any],
    scheduled_at: float,
    scheduled_offset_s: float,
) -> dict[str, Any]:
    await asyncio.sleep(max(0.0, scheduled_at - time.monotonic()))
    request_id = str(row["request_id"])
    ready_monotonic = time.monotonic()
    async with semaphore:
        send_monotonic = time.monotonic()
        send_time = time.time()
        first_token_monotonic: float | None = None
        completion_monotonic: float | None = None
        response_headers_monotonic: float | None = None
        completion_tokens = 0
        finish_reason = ""
        error = ""
        response_chars = 0
        max_tokens = requested_output_tokens(row, args.max_tokens_cap)
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": row["prompt"],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if args.ignore_eos:
            payload["ignore_eos"] = True
        try:
            headers = {
                "x-request-id": request_id,
                "x-pd-stage-timing": "1",
                "x-pd-client-send-time": f"{send_time:.9f}",
            }
            async with client.stream(
                "POST",
                f"{args.base_url.rstrip('/')}/v1/completions",
                json=payload,
                headers=headers,
                timeout=args.timeout_s,
            ) as response:
                response_headers_monotonic = time.monotonic()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        completion_monotonic = time.monotonic()
                        continue
                    event = json.loads(data)
                    usage = event.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or []
                    if choices:
                        text = choices[0].get("text") or ""
                        if text and first_token_monotonic is None:
                            first_token_monotonic = time.monotonic()
                        response_chars += len(text)
                        if choices[0].get("finish_reason"):
                            finish_reason = str(choices[0]["finish_reason"])
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        finish_monotonic = completion_monotonic or time.monotonic()
        ttft = None if first_token_monotonic is None else first_token_monotonic - send_monotonic
        tpot = None
        if ttft is not None and completion_tokens > 1:
            tpot = (finish_monotonic - first_token_monotonic) / (completion_tokens - 1)
        return {
            "request_id": request_id,
            "trace_class": row.get("trace_class", ""),
            "phase": row.get("phase", ""),
            "stream_type": row.get("stream_type", ""),
            "input_tokens": row.get("input_tokens"),
            "target_output_tokens": max_tokens,
            "scheduled_offset_s": scheduled_offset_s,
            "client_queue_s": send_monotonic - ready_monotonic,
            "send_time": send_time,
            "response_headers_s": None if response_headers_monotonic is None else response_headers_monotonic - send_monotonic,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "e2e_s": finish_monotonic - send_monotonic,
            "completion_tokens": completion_tokens,
            "response_chars": response_chars,
            "finish_reason": finish_reason,
            "success": not error,
            "error": error,
        }


def summarize(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]
    result: dict[str, Any] = {
        "submitted": len(rows),
        "completed": len(successful),
        "failed": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows),
        "wall_s": wall_s,
        "request_throughput": len(successful) / wall_s,
        "token_throughput": sum(row["completion_tokens"] for row in successful) / wall_s,
        "target_tokens": sum(row["target_output_tokens"] for row in rows),
    }
    for key in ("client_queue_s", "response_headers_s", "ttft_s", "tpot_s", "e2e_s"):
        values = [float(row[key]) for row in successful if row[key] is not None]
        result[key] = {
            "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }
    return result


async def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_trace(args.trace)
    if all(row.get("arrival_offset_s") is not None for row in rows):
        offsets = [float(row["arrival_offset_s"]) for row in rows]
        if offsets != sorted(offsets):
            raise ValueError("Trace arrival_offset_s values must be sorted")
    else:
        offsets = [index / args.request_rate for index in range(len(rows))]
    start = time.monotonic()
    limits = httpx.Limits(
        max_connections=args.max_concurrency,
        max_keepalive_connections=min(args.max_concurrency, 512),
    )
    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(args.max_concurrency)
        tasks = [
            asyncio.create_task(
                run_request(client, semaphore, args, row, start + offset, offset)
            )
            for row, offset in zip(rows, offsets)
        ]
        results = await asyncio.gather(*tasks)
    wall_s = time.monotonic() - start
    return results, {"arrival_offsets": offsets, "summary": summarize(results, wall_s)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8300")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens-cap", type=int, default=32768)
    parser.add_argument("--max-concurrency", type=int, default=2048)
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, metadata = asyncio.run(run(args))
    with (args.output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "arrival-schedule.json").write_text(
        json.dumps(metadata["arrival_offsets"], indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata["summary"], indent=2), encoding="utf-8"
    )
    if metadata["summary"]["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
