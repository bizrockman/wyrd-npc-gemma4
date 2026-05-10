"""
Multi-run comparison: run each suite N times against each model+spec
combination to measure pass-rate consistency and detect temperature variance.

Usage:
    python test_multirun.py
    python test_multirun.py --runs 3
    python test_multirun.py --suites natural_visit,role_competence
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT / "bench" / "results"

# (model_key, character_file_relative, label)
TARGETS = [
    ("gemma4:e2b", "characters/kenji_sato.dialogue_only.en.yaml", "e2b dialogue-only"),
    ("gemma4:e4b", "characters/kenji_sato.en.yaml", "e4b scene+dialogue"),
    ("gemma4:26b", "characters/kenji_sato.en.yaml", "26b scene+dialogue"),
    ("gemma4:31b", "characters/kenji_sato.en.yaml", "31b scene+dialogue"),
]

DEFAULT_SUITES = ["natural_visit", "role_competence"]


def run_suite(model: str, character_file: str, suite: str) -> dict | None:
    """Run a single suite against a single model. Returns the result dict
    or None on failure."""
    cmd = [
        sys.executable, "-X", "utf8",
        str(PROJECT / "bench" / "run_suite.py"),
        "--suite", f"kenji_sato_{suite}",
        "--models", model,
        "--character-file", character_file,
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8")

    if proc.returncode not in (0, 1):  # 1 = some failures, 0 = all pass
        print(f"  ERROR: {proc.stderr[:500]}")
        return None

    # Find the result file from stdout
    for line in proc.stdout.splitlines():
        if line.startswith("Results saved to:"):
            path = line.split(":", 1)[1].strip()
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
    return None


def aggregate_runs(runs: list[dict]) -> dict:
    """Aggregate texture and pass-rate metrics across multiple runs."""
    if not runs:
        return {}

    pass_counts = []
    hard_failure_counts = []
    expected_failure_counts = []
    question_shape_counts = []
    mean_words = []
    mean_dialogue = []
    mean_scene = []
    mean_latency = []
    scenario_passes = defaultdict(int)
    scenario_total = defaultdict(int)

    for run in runs:
        for model_data in run["models"]:
            tx = model_data.get("texture", {})
            pass_counts.append(model_data["passed"])

            # Count failures across all turns
            hard_failures = 0
            expected_failures = 0
            question_shape = 0
            for sc in model_data["scenarios"]:
                scenario_total[sc["scenario_id"]] += 1
                if not sc["hard_failure"]:
                    scenario_passes[sc["scenario_id"]] += 1
                for t in sc["turns"]:
                    for flag in t.get("hard_failures", []):
                        if flag.startswith("expected_hard_failure:"):
                            expected_failures += 1
                        elif flag.startswith("question_shape:"):
                            question_shape += 1
                        else:
                            hard_failures += 1
            hard_failure_counts.append(hard_failures)
            expected_failure_counts.append(expected_failures)
            question_shape_counts.append(question_shape)

            mean_words.append(tx.get("mean_words", 0))
            mean_latency.append(tx.get("mean_latency_s", 0))

    n = len(runs)
    total_scenarios = len(runs[0]["models"][0]["scenarios"])
    return {
        "runs": n,
        "scenarios_per_run": total_scenarios,
        "pass_rate": sum(pass_counts) / (n * total_scenarios) if n else 0,
        "scenario_pass_rate": {
            sid: scenario_passes[sid] / scenario_total[sid]
            for sid in scenario_total
        },
        "avg_hard_failures": sum(hard_failure_counts) / n if n else 0,
        "avg_expected_failures": sum(expected_failure_counts) / n if n else 0,
        "avg_question_shape_failures": sum(question_shape_counts) / n if n else 0,
        "mean_words": sum(mean_words) / n if n else 0,
        "mean_latency_s": sum(mean_latency) / n if n else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per combination")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--targets", default=None,
                        help="Comma-separated model names to filter (e.g., gemma4:e2b,gemma4:e4b)")
    args = parser.parse_args()

    suites = args.suites.split(",")
    targets = TARGETS
    if args.targets:
        wanted = set(args.targets.split(","))
        targets = [t for t in TARGETS if t[0] in wanted]

    print(f"Multi-run comparison")
    print(f"  Runs per cell: {args.runs}")
    print(f"  Suites: {suites}")
    print(f"  Targets: {[t[2] for t in targets]}")
    total_runs = args.runs * len(targets) * len(suites)
    print(f"  Total suite runs: {total_runs}")
    print("=" * 80)

    aggregated = {}  # (label, suite) -> aggregated metrics
    start_time = time.time()

    for model, char_file, label in targets:
        for suite in suites:
            print(f"\n[{label}] suite={suite}")
            runs = []
            for i in range(args.runs):
                t0 = time.time()
                result = run_suite(model, char_file, suite)
                elapsed = time.time() - t0
                if result is None:
                    print(f"  Run {i+1}: FAILED")
                    continue
                m = result["models"][0]
                tx = m.get("texture", {})
                print(f"  Run {i+1}: {m['passed']}/{m['total']} passed, "
                      f"{tx.get('mean_words', 0):.1f}w avg, {elapsed:.1f}s")
                runs.append(result)

            aggregated[(label, suite)] = aggregate_runs(runs)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Total time: {elapsed/60:.1f} min")
    print("=" * 80)

    # Print comparison table
    print("\nAGGREGATED RESULTS")
    print("=" * 110)
    print(f"  {'target':<25}{'suite':<18}{'pass%':>7}{'hard':>6}{'exp':>5}{'qshape':>8}{'words':>7}{'lat':>6}")
    print(f"  {'-'*25}{'-'*18}{'-'*7}{'-'*6}{'-'*5}{'-'*8}{'-'*7}{'-'*6}")
    for (label, suite), agg in aggregated.items():
        pass_pct = agg["pass_rate"] * 100
        print(f"  {label:<25}{suite:<18}"
              f"{pass_pct:>6.0f}%"
              f"{agg['avg_hard_failures']:>6.1f}"
              f"{agg['avg_expected_failures']:>5.1f}"
              f"{agg['avg_question_shape_failures']:>8.1f}"
              f"{agg['mean_words']:>7.1f}"
              f"{agg['mean_latency_s']:>6.1f}")

    # Per-scenario breakdown
    print("\nPER-SCENARIO PASS RATES")
    print("=" * 110)
    for (label, suite), agg in aggregated.items():
        if not agg.get("scenario_pass_rate"):
            continue
        print(f"\n  {label} / {suite}:")
        for sid, rate in sorted(agg["scenario_pass_rate"].items()):
            bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
            print(f"    {sid}: {bar} {rate*100:.0f}%")

    # Save full aggregate
    out = PROJECT / "bench" / "results" / f"multirun_{int(time.time())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"runs": args.runs, "suites": suites,
                       "targets": [t[2] for t in targets]},
            "aggregated": {f"{label}|{suite}": agg
                           for (label, suite), agg in aggregated.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nAggregate saved to: {out}")


if __name__ == "__main__":
    main()
