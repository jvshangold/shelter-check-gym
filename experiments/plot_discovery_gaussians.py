"""Plot Gaussian discovery-time distributions for each loophole."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.plot_loophole_metrics import read_csv_rows  # noqa: E402


def gaussian_pdf(x: float, mean: float, stddev: float) -> float:
    variance = stddev ** 2
    return (
        math.exp(-((x - mean) ** 2) / (2.0 * variance))
        / math.sqrt(2.0 * math.pi * variance)
    )


def distribution_rows(runs: list[dict]) -> list[dict]:
    grouped = {}
    for run in runs:
        steps = run.get("steps_to_first_discovery")
        if not isinstance(steps, (int, float)):
            continue
        grouped.setdefault(run["loophole"], []).append(float(steps))

    rows = []
    for loophole, steps in grouped.items():
        if len(steps) == 1:
            stddev = max(1.0, steps[0] * 0.05)
        else:
            stddev = statistics.stdev(steps)
            if stddev == 0.0:
                stddev = max(1.0, statistics.fmean(steps) * 0.05)
        rows.append(
            {
                "loophole": loophole,
                "steps": steps,
                "mean": statistics.fmean(steps),
                "median": statistics.median(steps),
                "stddev": stddev,
            }
        )

    return sorted(rows, key=lambda row: row["median"])


def write_discovery_gaussians(
    result_dir: Path,
    output_path: Path,
    points: int,
) -> None:
    runs_path = result_dir / "runs.csv"
    if not runs_path.exists():
        raise SystemExit(f"{result_dir} must contain runs.csv")

    distributions = distribution_rows(read_csv_rows(runs_path))
    if not distributions:
        raise SystemExit("No discovered runs with steps_to_first_discovery found.")

    max_x = max(
        max(row["steps"]) + 3.0 * row["stddev"]
        for row in distributions
    )
    max_x = max(max_x, 1.0)
    x_values = [
        max_x * i / max(points - 1, 1)
        for i in range(points)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(12, 6))
    for row in distributions:
        y_values = [
            gaussian_pdf(x, row["mean"], row["stddev"])
            for x in x_values
        ]
        peak = max(y_values) if y_values else 1.0
        scaled_y = [y / peak for y in y_values]
        plt.plot(
            x_values,
            scaled_y,
            linewidth=2,
            label=(
                f"{row['loophole']} "
                f"mean={row['mean']:.0f} std={row['stddev']:.0f}"
            ),
        )
        plt.axvline(
            row["mean"],
            linestyle=":",
            linewidth=1,
            alpha=0.35,
        )

    plt.xlabel("Steps to first discovery")
    plt.ylabel("Relative density")
    plt.title("Discovery-time distributions by loophole")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=Path("experiments/results/loophole_learning_metrics_combined"),
        help="Metrics directory containing runs.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "experiments/results/loophole_learning_metrics_combined/"
            "steps_to_first_discovery_gaussians.png"
        ),
    )
    parser.add_argument("--points", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_discovery_gaussians(args.result_dir, args.output_path, args.points)
    print(f"wrote discovery Gaussian plot to {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
