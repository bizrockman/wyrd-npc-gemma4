"""
Matrix driver: run a model (or models) through multiple suites sequentially,
collect per-cell result paths, write a manifest.

Per-model: runs each suite in turn. Per-suite: invokes run_suite.py.
Automatically picks the right character spec for e2b (dialogue_only).

Usage:
    python bench/run_matrix.py --models sonnet-4.6
    python bench/run_matrix.py --models sonnet-4.6,gemma4:e4b --suites all
    python bench/run_matrix.py --models gemma4:e2b --suites tgo,nv,rc
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_DIR / "bench"
RESULTS_DIR = BENCH_DIR / "results"

# Suite shortname -> full suite_id (file basename without .yaml)
SUITE_ALIASES = {
    "core": "kenji_sato_core_six",
    "core_six": "kenji_sato_core_six",
    "stress": "kenji_sato_stress",
    "play": "kenji_sato_playability",
    "playability": "kenji_sato_playability",
    "bc": "kenji_sato_boundary_check",
    "boundary_check": "kenji_sato_boundary_check",
    "tgo": "kenji_sato_trust_gate_opening",
    "trust_gate_opening": "kenji_sato_trust_gate_opening",
    "nv": "kenji_sato_natural_visit",
    "natural_visit": "kenji_sato_natural_visit",
    "rc": "kenji_sato_role_competence",
    "role_competence": "kenji_sato_role_competence",
}

ALL_SUITES = [
    "kenji_sato_core_six",
    "kenji_sato_stress",
    "kenji_sato_playability",
    "kenji_sato_boundary_check",
    "kenji_sato_trust_gate_opening",
    "kenji_sato_natural_visit",
    "kenji_sato_role_competence",
]

# Some models have a deployment-mode spec different from the default
CHARACTER_FILE_OVERRIDES = {
    "gemma4:e2b": "characters/kenji_sato.dialogue_only.en.yaml",
}


def resolve_suite(name: str) -> str:
    if name in ALL_SUITES:
        return name
    return SUITE_ALIASES.get(name, f"kenji_sato_{name}")


def run_one(model: str, suite_name: str, prompt_template: str,
            character_file: str | None) -> tuple[str | None, float, str, str]:
    """Run one (model, suite) cell. Return (result_path, elapsed, score, summary)."""
    cmd = [
        sys.executable, "-X", "utf8",
        str(BENCH_DIR / "run_suite.py"),
        "--suite", suite_name,
        "--models", model,
        "--prompt-template", prompt_template,
    ]
    if character_file:
        cmd.extend(["--character-file", character_file])

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8")
    elapsed = time.time() - t0

    # Extract result path and score from stdout
    result_path = None
    score = "?"
    for line in proc.stdout.splitlines():
        if line.startswith("Results saved to:"):
            result_path = line.split(":", 1)[1].strip()
        m = re.match(r"\s*(\S+)\s+(\d+/\d+)\s+(PASS|FAIL|partial)", line)
        if m and m.group(1).startswith(model.split("/")[0]):
            score = m.group(2)

    summary = f"exit={proc.returncode}"
    if proc.returncode != 0:
        # Include tail of stderr/stdout for debugging
        tail = (proc.stderr or "")[-300:] or (proc.stdout or "")[-300:]
        summary += f" tail={tail!r}"

    return result_path, elapsed, score, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True,
                        help="Comma-separated model keys")
    parser.add_argument("--suites", default="all",
                        help="Comma-separated suite names (aliases ok) or 'all'")
    parser.add_argument("--prompt-template", default="framework_compact")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.suites == "all":
        suites = list(ALL_SUITES)
    else:
        suites = [resolve_suite(s.strip()) for s in args.suites.split(",")]

    total_cells = len(models) * len(suites)
    print(f"{'='*78}")
    print(f"Matrix: {len(models)} model(s) x {len(suites)} suite(s) = {total_cells} cells")
    print(f"Prompt template: {args.prompt_template}")
    print(f"Suites: {suites}")
    print(f"{'='*78}")

    matrix = {}
    grand_start = time.time()

    for mi, model in enumerate(models, 1):
        print(f"\n[Model {mi}/{len(models)}] {model}")
        character_file = CHARACTER_FILE_OVERRIDES.get(model)
        if character_file:
            print(f"  spec: {character_file}")

        matrix[model] = {}
        model_start = time.time()
        for si, suite in enumerate(suites, 1):
            print(f"  ({si}/{len(suites)}) {suite} ...", end=" ", flush=True)
            path, elapsed, score, summary = run_one(
                model, suite, args.prompt_template, character_file
            )
            matrix[model][suite] = {
                "score": score,
                "elapsed_s": round(elapsed, 1),
                "result_file": path,
                "summary": summary,
            }
            tag = score if path else "ERROR"
            print(f"{tag:>6s}  {elapsed:>5.0f}s")

        model_elapsed = time.time() - model_start
        print(f"  [model done] {model_elapsed/60:.1f} min")

    grand_elapsed = time.time() - grand_start
    print(f"\n{'='*78}")
    print(f"Matrix total: {grand_elapsed/60:.1f} min")
    print(f"{'='*78}")

    # Summary table
    print("\n  Model".ljust(22) + "".join(f"{s.replace('kenji_sato_', '')[:14]:>16s}" for s in suites))
    print("  " + "-" * (20 + 16 * len(suites)))
    for model in models:
        row = f"  {model[:20]:20s}"
        for suite in suites:
            cell = matrix[model][suite]
            row += f"{cell['score']:>16s}"
        print(row)

    # Save manifest
    ts = int(time.time())
    manifest_path = RESULTS_DIR / f"matrix_{ts}.json"
    manifest_path.write_text(json.dumps({
        "timestamp": ts,
        "prompt_template": args.prompt_template,
        "total_elapsed_min": round(grand_elapsed / 60, 1),
        "matrix": matrix,
    }, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
