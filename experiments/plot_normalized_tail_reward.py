"""Plot average tail reward normalized by each loophole's maximum reward."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.loophole_learning_metrics import max_reward_by_loophole  # noqa: E402
from experiments.plot_loophole_metrics import read_csv_rows  # noqa: E402


def write_normalized_tail_reward(result_dir: Path, output_path: Path) -> None:
    summary_path = result_dir / "summary.csv"
    update_metrics_path = result_dir / "update_metrics.csv"
    if not summary_path.exists() or not update_metrics_path.exists():
        raise SystemExit(
            f"{result_dir} must contain summary.csv and update_metrics.csv"
        )

    summaries = read_csv_rows(summary_path)
    update_rows = read_csv_rows(update_metrics_path)
    max_rewards = max_reward_by_loophole(update_rows)

    rows = []
    for summary in summaries:
        loophole = summary["loophole"]
        tail_reward = summary.get("avg_tail_reward_per_episode")
        if not isinstance(tail_reward, (int, float)):
            continue
        max_reward = max_rewards.get(loophole, 0.0)
        reward_scale = max_reward if max_reward > 0.0 else 1.0
        rows.append(
            {
                "loophole": loophole,
                "normalized_tail_reward": tail_reward / reward_scale,
                "tail_reward": tail_reward,
                "max_reward": max_reward,
            }
        )

    if not rows:
        raise SystemExit("No tail reward rows found.")

    rows.sort(key=lambda row: row["normalized_tail_reward"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [row["loophole"] for row in rows]
    values = [row["normalized_tail_reward"] for row in rows]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, values)
    plt.ylabel("Average tail reward / max reward for loophole")
    plt.title("Normalized average tail reward per episode")
    plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1.0)
    plt.xticks(rotation=25, ha="right")

    for bar, row in zip(bars, rows):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['normalized_tail_reward']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

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
        help="Metrics directory containing summary.csv and update_metrics.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "experiments/results/loophole_learning_metrics_combined/"
            "normalized_avg_tail_reward_per_episode.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_normalized_tail_reward(args.result_dir, args.output_path)
    print(f"wrote normalized tail reward plot to {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
