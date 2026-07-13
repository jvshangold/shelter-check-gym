"""Plot all Barnes runs on shared normalized reward-over-steps axes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.loophole_learning_metrics import parse_number  # noqa: E402


def parse_value(value: str):
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    return parse_number(value)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return [
            {key: parse_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def default_result_dirs() -> list[Path]:
    results_root = Path("experiments/results")
    return sorted(
        path
        for path in results_root.glob("loophole_learning_metrics_barnes_group_*")
        if (path / "update_metrics.csv").exists()
    )


def plot_barnes_runs(
    update_rows: list[dict],
    output_path: Path,
    metric: str = "avg_reward_per_episode",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    barnes_rows = [
        row for row in update_rows
        if row.get("loophole") == "barnes_group"
        and isinstance(row.get(metric), (int, float))
    ]
    if not barnes_rows:
        raise SystemExit("No barnes_group update metrics found.")

    max_reward = max(row[metric] for row in barnes_rows)
    reward_scale = max_reward if max_reward > 0.0 else 1.0

    rows_by_run = {}
    for row in barnes_rows:
        rows_by_run.setdefault(int(row["run"]), []).append(row)

    plt.figure(figsize=(12, 6))
    for run, rows in sorted(rows_by_run.items()):
        rows.sort(key=lambda row: row["total_steps"])
        seed = rows[0].get("seed")
        plt.plot(
            [row["total_steps"] for row in rows],
            [row[metric] / reward_scale for row in rows],
            linewidth=1.8,
            alpha=0.85,
            label=f"run={run} seed={seed}",
        )

        discovery_rows = [
            row for row in rows
            if isinstance(row.get("success_count"), (int, float))
            and row["success_count"] > 0
        ]
        if discovery_rows:
            first_discovery = min(discovery_rows, key=lambda row: row["total_steps"])
            plt.scatter(
                [first_discovery["total_steps"]],
                [first_discovery[metric] / reward_scale],
                color="red",
                s=22,
                zorder=3,
            )

    plt.xlabel("Environment steps")
    plt.ylabel(f"{metric} / max Barnes {metric}")
    plt.title("Barnes group runs: normalized reward over time")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dirs",
        nargs="*",
        type=Path,
        help="Barnes result directories, or combined directories with update_metrics.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("experiments/results/barnes_group_runs_normalized_reward.png"),
    )
    parser.add_argument("--metric", default="avg_reward_per_episode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = args.results_dirs or default_result_dirs()
    if not result_dirs:
        raise SystemExit("No Barnes result directories found.")

    update_rows = []
    for result_dir in result_dirs:
        update_metrics_path = result_dir / "update_metrics.csv"
        if not update_metrics_path.exists():
            print(f"skipping missing update metrics: {result_dir}", file=sys.stderr)
            continue
        update_rows.extend(read_csv_rows(update_metrics_path))

    plot_barnes_runs(update_rows, args.output_path, metric=args.metric)
    print(f"wrote Barnes run plot to {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
