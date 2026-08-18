from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/1p3d_burst_20260817"
COLORS = {
    "default_fcfs": "#4C78A8",
    "least_active": "#F58518",
    "token_balance": "#54A24B",
}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_recovery_s(metrics_path: Path, post_end_s: float) -> float:
    last_positive_s = post_end_s
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        if float(sample["elapsed_s"]) < post_end_s:
            continue
        waiting = sum(
            float(worker.get("waiting") or 0)
            for worker in sample.get("workers", [])
            if "error" not in worker
        )
        if waiting > 0:
            last_positive_s = float(sample["elapsed_s"])
    return last_positive_s - post_end_s


def overall_rows(
    summaries: list[dict[str, Any]], output_root: Path, post_end_s: float
) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "strategy": summary["strategy"],
                "submitted": summary["submitted"],
                "completed": summary["completed"],
                "failed": summary["failed"],
                "wall_s": summary["wall_s"],
                "token_throughput": summary["token_throughput"],
                "ttft_mean_s": summary["ttft_s"]["mean"],
                "ttft_p99_s": summary["ttft_s"]["p99"],
                "tpot_mean_s": summary["tpot_s"]["mean"],
                "tpot_p99_s": summary["tpot_s"]["p99"],
                "e2e_mean_s": summary["e2e_s"]["mean"],
                "e2e_p99_s": summary["e2e_s"]["p99"],
                "waiting_request_seconds": summary["overall_metrics"]["waiting_request_seconds_approx"],
                "preemptions": sum(
                    worker["preemptions"] for worker in summary["overall_metrics"]["workers"].values()
                ),
                "first_zero_waiting_after_post_s": summary["phase_metrics"]["recovery_to_zero_waiting_s"],
                "stable_zero_waiting_after_post_s": stable_recovery_s(
                    output_root / "formal" / summary["strategy"] / "run/metrics.jsonl", post_end_s
                ),
            }
        )
    return rows


def phase_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for phase in ("pre", "burst", "post"):
            latency = summary["phase_latency"][phase]
            mechanism = summary["phase_metrics"][phase]
            rows.append(
                {
                    "strategy": summary["strategy"],
                    "phase": phase,
                    "completed": latency["completed"],
                    "ttft_mean_s": latency["ttft_s"]["mean"],
                    "ttft_p99_s": latency["ttft_s"]["p99"],
                    "tpot_mean_s": latency["tpot_s"]["mean"],
                    "tpot_p99_s": latency["tpot_s"]["p99"],
                    "e2e_mean_s": latency["e2e_s"]["mean"],
                    "e2e_p99_s": latency["e2e_s"]["p99"],
                    "peak_kv": mechanism["peak_kv"],
                    "peak_waiting": mechanism["peak_waiting"],
                    "waiting_request_seconds": mechanism["waiting_request_seconds_approx"],
                    "preemptions_delta": mechanism["preemptions_delta"],
                }
            )
    return rows


def assignment_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for phase, workers in summary["assignments"].items():
            for worker, values in workers.items():
                rows.append({"strategy": summary["strategy"], "phase": phase, "worker": worker, **values})
    return rows


def load_metric_series(path: Path) -> list[dict[str, Any]]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        workers = [worker for worker in sample.get("workers", []) if "error" not in worker]
        samples.append(
            {
                "elapsed_s": float(sample["elapsed_s"]),
                "waiting": sum(float(worker.get("waiting") or 0) for worker in workers),
                "kv_mean": sum(float(worker.get("kv") or 0) for worker in workers) / len(workers),
            }
        )
    return samples


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def line_chart(
    series: dict[str, list[dict[str, Any]]],
    key: str,
    ylabel: str,
    boundaries: dict[str, dict[str, float]],
    output: Path,
) -> None:
    width, height = 1400, 760
    left, right, top, bottom = 100, 40, 105, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = font(28), font(20), font(16)
    max_x = max(points[-1]["elapsed_s"] for points in series.values())
    max_y = max(point[key] for points in series.values() for point in points) or 1.0

    def xy(x: float, y: float) -> tuple[float, float]:
        px = left + x / max_x * (width - left - right)
        py = height - bottom - y / max_y * (height - top - bottom)
        return px, py

    draw.text((left, 12), f"1P3D burst experiment: {ylabel}", fill="black", font=title_font)
    draw.text((left, 65), ylabel, fill="black", font=label_font)
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    for index in range(6):
        value = max_y * index / 5
        y = xy(0, value)[1]
        draw.line((left, y, width - right, y), fill="#DDDDDD", width=1)
        draw.text((8, y - 10), f"{value:.0f}" if max_y > 10 else f"{value:.2f}", fill="black", font=small_font)
    for index in range(7):
        value = max_x * index / 6
        x = xy(value, 0)[0]
        draw.text((x - 20, height - bottom + 15), f"{value:.0f}", fill="black", font=small_font)
    for phase in ("pre", "burst", "post"):
        x = xy(float(boundaries[phase]["start_s"]), 0)[0]
        draw.line((x, top, x, height - bottom), fill="#888888", width=2)
        draw.text((x + 5, top + 5), phase, fill="#555555", font=small_font)
    for strategy, points in series.items():
        sampled = points[:: max(1, len(points) // 1200)]
        coords = [xy(point["elapsed_s"], point[key]) for point in sampled]
        if len(coords) > 1:
            draw.line(coords, fill=COLORS[strategy], width=4)
    legend_x = width - 310
    for index, strategy in enumerate(series):
        y = 28 + index * 25
        draw.line((legend_x, y + 8, legend_x + 35, y + 8), fill=COLORS[strategy], width=5)
        draw.text((legend_x + 45, y), strategy, fill="black", font=small_font)
    draw.text((width // 2 - 70, height - 40), "elapsed time (s)", fill="black", font=label_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    output_root = DEFAULT_OUTPUT
    summaries = json.loads((output_root / "formal-summary.json").read_text(encoding="utf-8"))
    if len(summaries) != 3:
        raise RuntimeError(f"Expected 3 completed strategies, found {len(summaries)}")
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    metadata = json.loads((output_root / "traces/burst-trace-metadata.json").read_text(encoding="utf-8"))
    overall = overall_rows(summaries, output_root, float(metadata["phases"]["post"]["end_s"]))
    phases = phase_rows(summaries)
    assignments = assignment_rows(summaries)
    write_csv(table_dir / "overall.csv", overall, list(overall[0]))
    write_csv(table_dir / "phase-latency-and-pressure.csv", phases, list(phases[0]))
    write_csv(table_dir / "assignments.csv", assignments, list(assignments[0]))
    series = {
        summary["strategy"]: load_metric_series(
            output_root / "formal" / summary["strategy"] / "run/metrics.jsonl"
        )
        for summary in summaries
    }
    line_chart(series, "waiting", "total waiting requests", metadata["phases"], figure_dir / "waiting-over-time.png")
    line_chart(series, "kv_mean", "mean Decode KV utilization", metadata["phases"], figure_dir / "kv-over-time.png")


if __name__ == "__main__":
    main()
