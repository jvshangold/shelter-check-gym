"""Write a clipped normalized reward-over-steps overlay from combined metrics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.loophole_learning_metrics import (  # noqa: E402
    max_reward_by_loophole,
    median_discovery_steps_by_loophole,
    representative_results,
)
from experiments.plot_loophole_metrics import read_csv_rows  # noqa: E402


def write_clipped_overlay(
    result_dir: Path,
    output_path: Path,
    max_steps: int,
) -> None:
    runs_path = result_dir / "runs.csv"
    update_metrics_path = result_dir / "update_metrics.csv"
    if not runs_path.exists() or not update_metrics_path.exists():
        raise SystemExit(
            f"{result_dir} must contain runs.csv and update_metrics.csv"
        )

    runs = read_csv_rows(runs_path)
    update_rows = read_csv_rows(update_metrics_path)
    selected = representative_results(runs)
    if not selected:
        raise SystemExit("No representative runs found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    max_rewards = max_reward_by_loophole(update_rows)
    median_lines = median_discovery_steps_by_loophole(runs)

    plt.figure(figsize=(12, 6))
    for loophole, result in selected.items():
        rows = [
            row for row in update_rows
            if row["loophole"] == loophole
            and row["run"] == result["run"]
            and row["total_steps"] <= max_steps
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: row["total_steps"])
        max_reward = max_rewards.get(loophole, 0.0)
        reward_scale = max_reward if max_reward > 0.0 else 1.0
        plt.plot(
            [row["total_steps"] for row in rows],
            [row["avg_reward_per_episode"] / reward_scale for row in rows],
            linewidth=2,
            label=f"{loophole} run={result['run']}",
        )

    for i, step in enumerate(
        sorted({step for step in median_lines.values() if step <= max_steps})
    ):
        plt.axvline(
            step,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            label="Median discovery step" if i == 0 else None,
        )

    plt.xlim(0, max_steps)
    plt.xlabel("Environment steps")
    plt.ylabel("Reward per episode / max reward for loophole")
    plt.title(f"Representative normalized reward curves, clipped at {max_steps} steps")
    plt.legend()
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
        help="Combined metrics directory containing runs.csv and update_metrics.csv.",
    )
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "experiments/results/loophole_learning_metrics_combined/"
            "representative_normalized_reward_per_episode_overlay_6000_steps.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_clipped_overlay(args.result_dir, args.output_path, args.max_steps)
    print(f"wrote clipped normalized reward plot to {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
