"""
Aggregate a matrix manifest into a single per-model row for the README.

For each model in the manifest:
  - Sums total_turns, total_dialogue_words, total_visible_words, total_latency_s
    across all per-suite result files.
  - Reports per-turn means: dw/t = total_dialogue_words / total_turns,
    w/t = total_visible_words / total_turns, lat/t = total_latency_s / total_turns.
  - Surfaces per-suite scores (passed/total) from each result file directly,
    so the scores don't depend on whatever run_suite printed in stdout.

Per-turn here means: mean over every single produced message across every
scenario. A 30-turn Play dialogue contributes 30 messages; a 2-turn BC probe
contributes 2. No per-dialog rollup before averaging.

Usage:
    python bench/aggregate_matrix.py bench/results/matrix_<ts>.json
"""
import argparse
import json
import sys
from pathlib import Path

import yaml


def load_suite_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_block_from_suite(suite_data, model_key):
    """run_suite writes one record with a 'models' list, each entry tagged by 'model'."""
    for m in suite_data.get("models", []):
        if m.get("model") == model_key:
            return m
    # Fall back to first entry if only one was run.
    models = suite_data.get("models", [])
    if len(models) == 1:
        return models[0]
    return None


def aggregate_model(model_key, suite_cells):
    """suite_cells: dict suite_name -> matrix cell (with result_file)."""
    total_turns = 0
    total_dlg = 0
    total_vis = 0
    total_lat = 0.0
    per_suite = {}
    missing = []

    for suite_name, cell in suite_cells.items():
        path = cell.get("result_file")
        if not path or not Path(path).exists():
            missing.append(suite_name)
            per_suite[suite_name] = {"score": cell.get("score", "?"), "turns": 0}
            continue

        data = load_suite_result(path)
        mb = model_block_from_suite(data, model_key)
        if mb is None:
            missing.append(suite_name)
            per_suite[suite_name] = {"score": cell.get("score", "?"), "turns": 0}
            continue

        tx = mb.get("texture", {}) or {}
        turns = tx.get("total_turns", 0)
        dlg = tx.get("total_dialogue_words", 0)
        vis = tx.get("total_visible_words", 0)
        lat = tx.get("total_latency_s", 0.0)

        # Older result files predate the totals fields. Reconstruct by
        # walking turns directly when available, else fall back to means*turns.
        if turns and not dlg:
            dlg_sum = 0
            have_dlg = False
            for sc in mb.get("scenarios", []):
                for t in sc.get("turns", []):
                    if "dialogue_word_count" in t:
                        dlg_sum += t["dialogue_word_count"]
                        have_dlg = True
            if have_dlg:
                dlg = dlg_sum
            elif tx.get("mean_dialogue_words") is not None:
                dlg = int(round(tx["mean_dialogue_words"] * turns))
        if turns and not vis:
            vis_sum = 0
            have_vis = False
            for sc in mb.get("scenarios", []):
                for t in sc.get("turns", []):
                    if "word_count" in t:
                        vis_sum += t["word_count"]
                        have_vis = True
            if have_vis:
                vis = vis_sum
            elif tx.get("mean_words") is not None:
                vis = int(round(tx["mean_words"] * turns))
        if turns and not lat:
            lat_sum = 0.0
            have_lat = False
            for sc in mb.get("scenarios", []):
                for t in sc.get("turns", []):
                    if "latency_s" in t:
                        lat_sum += t["latency_s"]
                        have_lat = True
            if have_lat:
                lat = lat_sum
            elif tx.get("mean_latency_s") is not None:
                lat = tx["mean_latency_s"] * turns

        total_turns += turns
        total_dlg += dlg
        total_vis += vis
        total_lat += lat

        passed = mb.get("passed", 0)
        total = mb.get("total", 0)
        per_suite[suite_name] = {
            "score": f"{passed}/{total}" if total else cell.get("score", "?"),
            "turns": turns,
            "mean_dlg_words": tx.get("mean_dialogue_words"),
            "mean_words": tx.get("mean_words"),
            "mean_latency_s": tx.get("mean_latency_s"),
            "hard_failures": sum(1 for s in mb.get("scenarios", []) if s.get("hard_failure")),
        }

    out = {
        "model": model_key,
        "total_turns": total_turns,
        "total_dialogue_words": total_dlg,
        "total_visible_words": total_vis,
        "total_latency_s": round(total_lat, 1),
        "mean_dialogue_words_per_turn": round(total_dlg / total_turns, 2) if total_turns else None,
        "mean_visible_words_per_turn": round(total_vis / total_turns, 2) if total_turns else None,
        "mean_latency_s_per_turn": round(total_lat / total_turns, 2) if total_turns else None,
        "per_suite": per_suite,
        "missing_suites": missing,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of table")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    aggregates = []
    for model_key, suite_cells in manifest["matrix"].items():
        aggregates.append(aggregate_model(model_key, suite_cells))

    if args.json:
        print(json.dumps({"manifest": args.manifest, "models": aggregates}, indent=2))
        return

    # Pretty table
    suites_order = list(next(iter(manifest["matrix"].values())).keys())
    short = lambda s: s.replace("kenji_sato_", "")[:10]

    header = f"{'model':22s}  {'turns':>6s}  {'dw/t':>6s}  {'w/t':>6s}  {'lat/t':>6s}  "
    header += "  ".join(f"{short(s):>10s}" for s in suites_order)
    print(header)
    print("-" * len(header))
    for agg in aggregates:
        row = f"{agg['model']:22s}"
        row += f"  {agg['total_turns']:6d}"
        row += f"  {agg['mean_dialogue_words_per_turn'] or 0:6.1f}"
        row += f"  {agg['mean_visible_words_per_turn'] or 0:6.1f}"
        row += f"  {agg['mean_latency_s_per_turn'] or 0:6.2f}"
        for s in suites_order:
            cell = agg["per_suite"].get(s, {})
            row += f"  {cell.get('score', '?'):>10s}"
        print(row)
        if agg["missing_suites"]:
            print(f"  ! missing/unreadable: {agg['missing_suites']}")


if __name__ == "__main__":
    main()
