from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/1p3d_microburst_20260817"
STRATEGIES = ("default_fcfs", "least_active", "token_balance")
COLORS = {
    "default_fcfs": "#4C78A8",
    "least_active": "#F58518",
    "token_balance": "#54A24B",
}
KV_CAPACITY = 126576
INPUT_TOKENS = 131


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = json.loads((OUTPUT_ROOT / "formal-summary.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (OUTPUT_ROOT / "traces/formal_1200_background_21cycles.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if [summary["strategy"] for summary in summaries] != list(STRATEGIES):
        raise RuntimeError("Formal strategies are missing or out of order")
    return summaries, metadata


def stable_times(strategy: str, last_arrival_s: float) -> dict[str, float]:
    samples = [
        json.loads(line)
        for line in (
            OUTPUT_ROOT / f"formal/{strategy}/run/metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    last_waiting = max(
        (
            float(sample["elapsed_s"])
            for sample in samples
            if sum(
                float(worker.get("waiting") or 0)
                for worker in sample.get("workers", [])
                if "error" not in worker
            )
            > 0
        ),
        default=last_arrival_s,
    )
    last_active = max(
        (
            float(sample["elapsed_s"])
            for sample in samples
            if sample.get("router", {}).get("active_decode_reservations", 0) > 0
        ),
        default=last_arrival_s,
    )
    return {
        "last_waiting_after_arrival_s": max(0.0, last_waiting - last_arrival_s),
        "last_active_sample_after_arrival_s": max(0.0, last_active - last_arrival_s),
    }


def overall_rows(summaries: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    last_arrival = float(metadata["last_arrival_offset_s"])
    for summary in summaries:
        times = stable_times(summary["strategy"], last_arrival)
        rows.append(
            {
                "strategy": summary["strategy"],
                "submitted": summary["submitted"],
                "completed": summary["completed"],
                "failed": summary["failed"],
                "wall_s": summary["wall_s"],
                "drain_after_last_arrival_s": summary["wall_s"] - last_arrival,
                **times,
                "token_throughput": summary["token_throughput"],
                "ttft_mean_s": summary["ttft_s"]["mean"],
                "ttft_p99_s": summary["ttft_s"]["p99"],
                "tpot_mean_ms": summary["tpot_s"]["mean"] * 1000,
                "tpot_p99_ms": summary["tpot_s"]["p99"] * 1000,
                "e2e_mean_s": summary["e2e_s"]["mean"],
                "e2e_p99_s": summary["e2e_s"]["p99"],
                "waiting_request_seconds": summary["metrics"]["waiting_request_seconds_approx"],
                "request_preemptions": summary["preemptions"]["total_preemptions"],
                "recompute_tokens_approx": summary["preemptions"]["total_recompute_tokens_approx"],
            }
        )
    return rows


def assignment_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        outputs = [worker["output_tokens"] for worker in summary["routes"]["assignments"].values()]
        coefficient = statistics.pstdev(outputs) / statistics.mean(outputs)
        for worker, values in summary["routes"]["assignments"].items():
            estimated_kv = values["output_tokens"] + values["requests"] * INPUT_TOKENS
            rows.append(
                {
                    "strategy": summary["strategy"],
                    "worker": worker,
                    **values,
                    "estimated_burst_kv_tokens": estimated_kv,
                    "estimated_burst_kv_fraction": estimated_kv / KV_CAPACITY,
                    "strategy_output_token_cv": coefficient,
                    "strict_completed_gate": summary["routes"]["strict_completed_gate"],
                    "completed_snapshot": "/".join(
                        str(value) for value in summary["routes"]["first_completed_snapshot"]
                    ),
                }
            )
    return rows


def latency_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for request_class, metrics in summary["latency_classes"].items():
            rows.append(
                {
                    "strategy": summary["strategy"],
                    "request_class": request_class,
                    "completed": metrics["completed"],
                    "ttft_mean_s": metrics["ttft_s"]["mean"],
                    "ttft_p99_s": metrics["ttft_s"]["p99"],
                    "tpot_mean_ms": metrics["tpot_s"]["mean"] * 1000,
                    "tpot_p99_ms": metrics["tpot_s"]["p99"] * 1000,
                    "e2e_mean_s": metrics["e2e_s"]["mean"],
                    "e2e_p99_s": metrics["e2e_s"]["p99"],
                }
            )
    return rows


def request_class(request_id: str) -> str:
    if request_id.startswith("background-"):
        return "background"
    index = int(request_id.split("-")[1])
    return "burst_long" if index % 3 == 0 else "burst_short"


def preemption_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        records = summary["preemptions"]["records"]
        for category in ("background", "burst_long", "burst_short"):
            selected = [record for record in records if request_class(record["request_id"]) == category]
            rows.append(
                {
                    "strategy": summary["strategy"],
                    "request_class": category,
                    "preemption_events": len(selected),
                    "unique_preempted_requests": len({record["request_id"] for record in selected}),
                    "recompute_tokens_approx": sum(record["computed_tokens"] for record in selected),
                    "worker_counts": json.dumps(Counter(record["worker"] for record in selected), sort_keys=True),
                }
            )
    return rows


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def waiting_plot(metadata: dict[str, Any]) -> None:
    width, height = 1400, 760
    left, right, top, bottom = 100, 40, 105, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    series: dict[str, list[tuple[float, float]]] = {}
    for strategy in STRATEGIES:
        points = []
        path = OUTPUT_ROOT / f"formal/{strategy}/run/metrics.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            waiting = sum(
                float(worker.get("waiting") or 0)
                for worker in sample.get("workers", [])
                if "error" not in worker
            )
            points.append((float(sample["elapsed_s"]), waiting))
        series[strategy] = points
    max_x = max(points[-1][0] for points in series.values())
    max_y = max(value for points in series.values() for _, value in points) or 1

    def xy(x: float, y: float) -> tuple[float, float]:
        return (
            left + x / max_x * (width - left - right),
            height - bottom - y / max_y * (height - top - bottom),
        )

    draw.text((left, 12), "1P3D heterogeneous microburst: total waiting requests", fill="black", font=font(28))
    draw.text((left, 65), "total waiting requests", fill="black", font=font(20))
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    for index in range(6):
        value = max_y * index / 5
        y = xy(0, value)[1]
        draw.line((left, y, width - right, y), fill="#DDDDDD", width=1)
        draw.text((30, y - 9), f"{value:.0f}", fill="black", font=font(15))
    burst_end_x = xy(float(metadata["burst_window_s"]), 0)[0]
    arrivals_end_x = xy(float(metadata["last_arrival_offset_s"]), 0)[0]
    for x, label in ((burst_end_x, "burst end"), (arrivals_end_x, "last arrival")):
        draw.line((x, top, x, height - bottom), fill="#888888", width=2)
        draw.text((x + 5, top + 5), label, fill="#555555", font=font(15))
    for strategy, points in series.items():
        draw.line([xy(x, y) for x, y in points], fill=COLORS[strategy], width=4)
    for index, strategy in enumerate(STRATEGIES):
        y = 28 + index * 25
        draw.line((1075, y + 8, 1110, y + 8), fill=COLORS[strategy], width=5)
        draw.text((1120, y), strategy, fill="black", font=font(15))
    draw.text((610, height - 40), "elapsed time (s)", fill="black", font=font(20))
    path = OUTPUT_ROOT / "figures/waiting-over-time.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def kv_bar_plot(assignments: list[dict[str, Any]]) -> None:
    width, height = 1400, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 100, 50, 100, 100
    max_y = max(row["estimated_burst_kv_fraction"] for row in assignments) * 1.1
    draw.text((left, 18), "Estimated burst KV demand by Decode worker", fill="black", font=font(30))
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    capacity_y = height - bottom - 1.0 / max_y * (height - top - bottom)
    draw.line((left, capacity_y, width - right, capacity_y), fill="#C62828", width=3)
    draw.text((width - 250, capacity_y - 25), "KV capacity 100%", fill="#C62828", font=font(17))
    group_width = (width - left - right) / len(STRATEGIES)
    worker_width = group_width / 5
    for strategy_index, strategy in enumerate(STRATEGIES):
        selected = [row for row in assignments if row["strategy"] == strategy]
        start = left + strategy_index * group_width + worker_width
        for worker_index, row in enumerate(selected):
            x0 = start + worker_index * worker_width
            x1 = x0 + worker_width * 0.72
            y0 = height - bottom - row["estimated_burst_kv_fraction"] / max_y * (height - top - bottom)
            draw.rectangle((x0, y0, x1, height - bottom), fill=COLORS[strategy])
            draw.text((x0, y0 - 23), f"{row['estimated_burst_kv_fraction'] * 100:.0f}%", fill="black", font=font(15))
            draw.text((x0 + 8, height - bottom + 8), row["worker"], fill="black", font=font(16))
        draw.text((start, height - 45), strategy, fill="black", font=font(18))
    path = OUTPUT_ROOT / "figures/burst-kv-by-worker.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    summaries, metadata = load()
    overall = overall_rows(summaries, metadata)
    assignments = assignment_rows(summaries)
    latencies = latency_rows(summaries)
    preemptions = preemption_rows(summaries)
    write_csv(OUTPUT_ROOT / "tables/overall.csv", overall)
    write_csv(OUTPUT_ROOT / "tables/burst-assignments.csv", assignments)
    write_csv(OUTPUT_ROOT / "tables/latency-by-class.csv", latencies)
    write_csv(OUTPUT_ROOT / "tables/request-preemptions.csv", preemptions)
    waiting_plot(metadata)
    kv_bar_plot(assignments)


if __name__ == "__main__":
    main()
