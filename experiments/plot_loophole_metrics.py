"""Combine per-loophole metrics outputs and write cross-loophole plots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.loophole_learning_metrics import (  # noqa: E402
    parse_number,
    summarize,
    write_csv,
    write_plots,
)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    key: parse_value(value)
                    for key, value in row.items()
                }
            )
    return rows


def parse_value(value: str):
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    return parse_number(value)


def default_result_dirs() -> list[Path]:
    results_root = Path("experiments/results")
    return sorted(
        path
        for path in results_root.glob("loophole_learning_metrics_*")
        if (path / "runs.csv").exists() and (path / "update_metrics.csv").exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dirs",
        nargs="*",
        type=Path,
        help="Result directories containing runs.csv and update_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/loophole_learning_metrics_combined"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = args.results_dirs or default_result_dirs()
    if not result_dirs:
        raise SystemExit("No result directories found.")

    runs = []
    update_rows = []
    for result_dir in result_dirs:
        runs_path = result_dir / "runs.csv"
        update_metrics_path = result_dir / "update_metrics.csv"
        if not runs_path.exists() or not update_metrics_path.exists():
            print(f"skipping incomplete result directory: {result_dir}", file=sys.stderr)
            continue
        runs.extend(read_csv_rows(runs_path))
        update_rows.extend(read_csv_rows(update_metrics_path))

    if not runs:
        raise SystemExit("No complete metrics rows found.")

    summaries = summarize(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", runs)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "update_metrics.csv", update_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "runs": runs,
                "summary": summaries,
                "update_metrics": update_rows,
            },
            handle,
            indent=2,
        )
    write_plots(args.output_dir, summaries, runs, update_rows)
    print(f"wrote combined metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
