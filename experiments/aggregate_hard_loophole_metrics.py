"""Combine no-mask hard-mode action-cost metrics outputs and write aggregate plots."""

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


DEFAULT_OUTPUT_DIR = Path(
    "experiments/results/loophole_learning_metrics_hard_nomask_actioncost_combined"
)


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


def default_result_dirs(results_root: Path, output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_root.glob("loophole_learning_metrics_hard_nomask_actioncost_*")
        if path != output_dir
        and (path / "runs.csv").exists()
        and (path / "update_metrics.csv").exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dirs",
        nargs="*",
        type=Path,
        help="Hard result directories containing runs.csv and update_metrics.csv.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/results"),
        help="Root directory to scan when no result directories are provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for combined hard metrics and aggregate plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = args.results_dirs or default_result_dirs(
        args.results_root,
        args.output_dir,
    )
    if not result_dirs:
        raise SystemExit(
            "No hard result directories found. Expected directories matching "
            "'experiments/results/loophole_learning_metrics_hard_nomask_actioncost_*'."
        )

    runs = []
    update_rows = []
    for result_dir in result_dirs:
        runs_path = result_dir / "runs.csv"
        update_metrics_path = result_dir / "update_metrics.csv"
        if not runs_path.exists() or not update_metrics_path.exists():
            print(f"skipping incomplete result directory: {result_dir}", file=sys.stderr)
            continue

        run_rows = read_csv_rows(runs_path)
        for row in run_rows:
            row.setdefault("env_variant", "hard")
            row.setdefault("source_dir", str(result_dir))
        runs.extend(run_rows)

        update_metric_rows = read_csv_rows(update_metrics_path)
        for row in update_metric_rows:
            row.setdefault("env_variant", "hard")
            row.setdefault("source_dir", str(result_dir))
        update_rows.extend(update_metric_rows)

    if not runs:
        raise SystemExit("No complete hard metrics rows found.")

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
                "source_dirs": [str(path) for path in result_dirs],
            },
            handle,
            indent=2,
        )

    write_plots(args.output_dir, summaries, runs, update_rows)
    print(f"combined {len(result_dirs)} hard result directories")
    print(f"wrote hard aggregate metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
