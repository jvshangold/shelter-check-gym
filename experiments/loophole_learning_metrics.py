"""Run repeated loophole-training probes and summarize discovery consistency.

The script imports each loophole's existing ``rl.train_agent`` module and calls
``train`` with snapshots disabled. It parses the normal training logs so the
training code remains the source of truth for rewards and success criteria.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LOOPHOLES = [
    "barnes_group",
    "distressed_assets",
    "dub_irish_dutch",
    "partnership_disguised_sale",
    "straddle_abuse",
    "subsidiary_stock_comp",
]

UPDATE_RE = re.compile(r"^(?:update_complete:|update:)\s*(?P<update>\d+);")
PAIR_RE = re.compile(r"(?P<key>[a-zA-Z_]+):\s*(?P<value>[^;]+)")


def parse_number(value: str):
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def parse_update_rows(output: str, rollout_steps: int) -> list[dict]:
    rows = []
    for line in output.splitlines():
        match = UPDATE_RE.match(line.strip())
        if not match:
            continue

        row = {"update": int(match.group("update"))}
        for pair in PAIR_RE.finditer(line):
            key = pair.group("key")
            if key in {"update", "update_complete"}:
                continue
            row[key] = parse_number(pair.group("value"))

        cumulative_reward = float(row.get("cumulative_reward", 0.0))
        episode_count = int(row.get("episode_count", 0))
        success_count = int(row.get("success_count", 0))
        row["total_steps"] = (row["update"] + 1) * rollout_steps
        row["avg_reward_per_episode"] = (
            cumulative_reward / episode_count if episode_count else cumulative_reward
        )
        row["success_rate"] = success_count / max(episode_count, 1)
        rows.append(row)
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_probe(loophole: str, run_index: int, seed: int, args) -> dict:
    seed_everything(seed)
    module = importlib.import_module(f"{loophole}.rl.train_agent")

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        module.train(
            total_updates=args.updates,
            rollout_steps=args.rollout_steps,
            save_snapshots=False,
            log_interval=1,
        )

    rows = parse_update_rows(captured.getvalue(), rollout_steps=args.rollout_steps)
    discovery_row = next((row for row in rows if int(row.get("success_count", 0)) > 0), None)
    post_discovery_rows = (
        [row for row in rows if row["update"] >= discovery_row["update"]]
        if discovery_row
        else []
    )
    tail_rows = rows[-args.tail_updates :] if rows else []

    return {
        "loophole": loophole,
        "run": run_index,
        "seed": seed,
        "discovered": discovery_row is not None,
        "discovery_update": discovery_row["update"] if discovery_row else None,
        "steps_to_first_discovery": discovery_row["total_steps"] if discovery_row else None,
        "post_discovery_avg_reward_per_episode": mean(
            row["avg_reward_per_episode"] for row in post_discovery_rows
        ),
        "tail_avg_reward_per_episode": mean(
            row["avg_reward_per_episode"] for row in tail_rows
        ),
        "tail_success_rate": mean(row["success_rate"] for row in tail_rows),
        "tail_reward_stddev": stdev(row["avg_reward_per_episode"] for row in tail_rows),
        "tail_success_rate_stddev": stdev(row["success_rate"] for row in tail_rows),
        "updates_logged": len(rows),
    }


def mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def stdev(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def coefficient_of_variation(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return None
    avg = statistics.fmean(values)
    if avg == 0:
        return None
    return statistics.stdev(values) / abs(avg)


def summarize(results: list[dict]) -> list[dict]:
    summaries = []
    loopholes = sorted({result["loophole"] for result in results})
    for loophole in loopholes:
        group = [result for result in results if result["loophole"] == loophole]
        discovered = [result for result in group if result["discovered"]]
        step_values = [result["steps_to_first_discovery"] for result in discovered]
        tail_rewards = [result["tail_avg_reward_per_episode"] for result in group]
        tail_success_rates = [result["tail_success_rate"] for result in group]
        summaries.append(
            {
                "loophole": loophole,
                "runs": len(group),
                "discovery_rate": len(discovered) / len(group),
                "avg_steps_to_first_discovery": mean(step_values),
                "steps_to_first_discovery_stddev": stdev(step_values),
                "steps_to_first_discovery_cv": coefficient_of_variation(step_values),
                "avg_tail_reward_per_episode": mean(tail_rewards),
                "tail_reward_stddev_across_runs": stdev(tail_rewards),
                "tail_reward_cv_across_runs": coefficient_of_variation(tail_rewards),
                "avg_tail_success_rate": mean(tail_success_rates),
                "tail_success_rate_stddev_across_runs": stdev(tail_success_rates),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, summaries: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["loophole"] for row in summaries]

    plot_specs = [
        (
            "avg_steps_to_first_discovery.png",
            "avg_steps_to_first_discovery",
            "Average steps to first discovery",
        ),
        (
            "avg_tail_reward_per_episode.png",
            "avg_tail_reward_per_episode",
            "Average reward per episode after learning",
        ),
        (
            "tail_success_rate_stddev.png",
            "tail_success_rate_stddev_across_runs",
            "Tail success-rate standard deviation",
        ),
    ]

    for filename, key, ylabel in plot_specs:
        values = [row.get(key) or 0.0 for row in summaries]
        plt.figure(figsize=(11, 5))
        plt.bar(labels, values)
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loopholes", nargs="+", default=DEFAULT_LOOPHOLES)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--tail-updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/loophole_learning_metrics"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for loophole in args.loopholes:
        for run_index in range(args.runs):
            seed = args.seed + run_index
            print(f"running {loophole} run={run_index} seed={seed}", flush=True)
            results.append(run_probe(loophole, run_index, seed, args))

    summaries = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", results)
    write_csv(args.output_dir / "summary.csv", summaries)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump({"runs": results, "summary": summaries}, handle, indent=2)
    write_plots(args.output_dir, summaries)
    print(f"wrote metrics to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
