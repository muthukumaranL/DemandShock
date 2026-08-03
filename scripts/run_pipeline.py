"""End-to-end pipeline: prepare -> train -> evaluate -> export.

Run:  python scripts/run_pipeline.py --mode development
      python scripts/run_pipeline.py --mode full
      python scripts/run_pipeline.py --mode full --smoke   # 1 arm, quick shape check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("prepare_data.py", "Ingest, validate and engineer features from real sources"),
    ("train.py", "Backtest, ablate, select, and fit the deploy model"),
    ("evaluate.py", "Score demand shocks and build the inventory base table"),
    ("export_results.py", "Regenerate RESULTS.md from the artifacts"),
]


def run(script: str, args: list[str]) -> int:
    command = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    print(f"\n$ {' '.join(command[1:])}\n", flush=True)
    return subprocess.call(command, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="DemandShock end-to-end pipeline")
    parser.add_argument("--mode", choices=["development", "full"], default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Reduced run for pipeline verification only. Results from "
                             "a smoke run must NOT be quoted as ablation findings.")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    common = ["--mode", args.mode] if args.mode else []
    started = time.time()

    for script, description in STEPS:
        if args.skip_prepare and script == "prepare_data.py":
            print(f"[skip] {script}")
            continue
        step_args = list(common)
        if script == "train.py" and args.smoke:
            step_args += ["--ablation-configs", "B", "--skip-objective-check"]
        if script == "export_results.py":
            step_args = list(common)
        print("=" * 72)
        print(f"STEP  {script} - {description}")
        print("=" * 72, flush=True)
        code = run(script, step_args)
        if code != 0:
            print(f"\nPipeline FAILED at {script} (exit {code}).", file=sys.stderr)
            return code

    minutes = (time.time() - started) / 60
    print("=" * 72)
    print(f"Pipeline complete in {minutes:.1f} min.")
    print("  streamlit run app/Home.py")
    print("  uvicorn api.main:app --reload")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
