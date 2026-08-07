"""Run repeated loophole-training probes and summarize discovery consistency.

The script imports each loophole's existing ``rl.train_agent`` module and calls
``train`` with snapshots disabled. It parses the normal training logs so the
training code remains the source of truth for rewards and success criteria.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime
import importlib
import io
import json
import os
import random
import re
import statistics
import sys
import time
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


class TeeCapture(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, value):
        self.stream.write(value)
        self.stream.flush()
        return super().write(value)


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


def run_probe(loophole: str, run_index: int, seed: int, args) -> tuple[dict, list[dict]]:
    seed_everything(seed)
    module = importlib.import_module(f"{loophole}.rl.train_agent")

    captured = TeeCapture(sys.stdout) if args.stream_train_logs else io.StringIO()
    started_at = time.perf_counter()
    with contextlib.redirect_stdout(captured):
        module.train(
            total_updates=args.updates,
            rollout_steps=args.rollout_steps,
            save_snapshots=False,
            log_interval=1,
            env_variant=args.env_variant,
        )
    elapsed_seconds = time.perf_counter() - started_at

    rows = parse_update_rows(captured.getvalue(), rollout_steps=args.rollout_steps)
    for row in rows:
        row["loophole"] = loophole
        row["run"] = run_index
        row["seed"] = seed

    discovery_row = next((row for row in rows if int(row.get("success_count", 0)) > 0), None)
    post_discovery_rows = (
        [row for row in rows if row["update"] >= discovery_row["update"]]
        if discovery_row
        else []
    )
    tail_rows = rows[-args.tail_updates :] if rows else []

    return {
        "loophole": loophole,
        "env_variant": args.env_variant,
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
        "elapsed_seconds": elapsed_seconds,
    }, rows


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


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def median(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return statistics.median(values) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_discovery_boxplot(output_dir: Path, results: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    discovered_by_loophole = {}
    for result in results:
        if result["steps_to_first_discovery"] is None:
            continue
        discovered_by_loophole.setdefault(result["loophole"], []).append(
            result["steps_to_first_discovery"]
        )

    ordered = sorted(
        discovered_by_loophole,
        key=lambda loophole: statistics.median(discovered_by_loophole[loophole]),
    )
    if not ordered:
        return

    labels = []
    values = []
    means = []
    for loophole in ordered:
        steps = discovered_by_loophole[loophole]
        labels.append(
            f"{loophole}\nmean={statistics.fmean(steps):.0f}\nstd={stdev(steps) or 0.0:.0f}"
        )
        values.append(steps)
        means.append(statistics.fmean(steps))

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        whis=(0, 100),
    )
    plt.plot(range(1, len(means) + 1), means, "o", color="black", label="Mean")
    plt.ylabel("Steps to first discovery")
    plt.title("Discovery time distribution by loophole")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "steps_to_first_discovery_boxplot.png", dpi=160)
    plt.close()


def representative_results(results: list[dict]) -> dict[str, dict]:
    selected = {}
    for loophole in sorted({result["loophole"] for result in results}):
        group = [result for result in results if result["loophole"] == loophole]
        discovered = [
            result for result in group
            if result["steps_to_first_discovery"] is not None
        ]
        if not discovered:
            selected[loophole] = group[0]
            continue

        median_steps = statistics.median(
            result["steps_to_first_discovery"] for result in discovered
        )
        selected[loophole] = min(
            discovered,
            key=lambda result: abs(result["steps_to_first_discovery"] - median_steps),
        )
    return selected


def median_discovery_steps_by_loophole(results: list[dict]) -> dict[str, float]:
    medians = {}
    for loophole in sorted({result["loophole"] for result in results}):
        steps = [
            result["steps_to_first_discovery"]
            for result in results
            if result["loophole"] == loophole
            and result["steps_to_first_discovery"] is not None
        ]
        if steps:
            medians[loophole] = statistics.median(steps)
    return medians


def max_reward_by_loophole(update_rows: list[dict]) -> dict[str, float]:
    max_rewards = {}
    for row in update_rows:
        reward = row.get("avg_reward_per_episode")
        if not isinstance(reward, (int, float)):
            continue
        loophole = row["loophole"]
        if loophole not in max_rewards or reward > max_rewards[loophole]:
            max_rewards[loophole] = reward
    return max_rewards


def write_reward_overlay_plot(
    output_dir: Path,
    results: list[dict],
    update_rows: list[dict],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    selected = representative_results(results)
    if not selected:
        return

    plt.figure(figsize=(12, 6))
    median_lines = median_discovery_steps_by_loophole(results)
    max_rewards = max_reward_by_loophole(update_rows)
    for loophole, result in selected.items():
        rows = [
            row for row in update_rows
            if row["loophole"] == loophole and row["run"] == result["run"]
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
            label=f"{loophole} run={result['run']} max={max_reward:.3g}",
        )

    for i, step in enumerate(sorted(set(median_lines.values()))):
        plt.axvline(
            step,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            label="Median discovery step" if i == 0 else None,
        )

    plt.xlabel("Environment steps")
    plt.ylabel("Reward per episode / max reward for loophole")
    plt.title("Representative normalized reward curves by loophole")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "representative_normalized_reward_per_episode_overlay.png",
        dpi=160,
    )
    plt.close()


def write_plots(
    output_dir: Path,
    summaries: list[dict],
    results: list[dict],
    update_rows: list[dict],
) -> None:
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

    write_discovery_boxplot(output_dir, results)
    write_reward_overlay_plot(output_dir, results, update_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loopholes", nargs="+", default=DEFAULT_LOOPHOLES)
    parser.add_argument(
        "--env-variant",
        choices=("easy", "hard"),
        default="easy",
        help="Select the scaffolded easy envs or the unscaffolded hard envs.",
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--tail-updates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--stream-train-logs",
        action="store_true",
        help="Mirror each trainer's update logs to stdout while still parsing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/loophole_learning_metrics"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    update_rows = []
    for loophole in args.loopholes:
        for run_index in range(args.runs):
            seed = args.seed + run_index
            print(
                f"{timestamp()} running {loophole} run={run_index} seed={seed}",
                flush=True,
            )
            result, run_update_rows = run_probe(loophole, run_index, seed, args)
            print(
                f"{timestamp()} completed {loophole} run={run_index} "
                f"elapsed_seconds={result['elapsed_seconds']:.2f} "
                f"updates_logged={result['updates_logged']}",
                flush=True,
            )
            results.append(result)
            update_rows.extend(run_update_rows)

    summaries = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", results)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "update_metrics.csv", update_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "runs": results,
                "summary": summaries,
                "update_metrics": update_rows,
            },
            handle,
            indent=2,
        )
    write_plots(args.output_dir, summaries, results, update_rows)
    print(f"wrote metrics to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
