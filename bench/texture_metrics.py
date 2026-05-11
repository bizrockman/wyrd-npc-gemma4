"""
Texture metrics — post-hoc deterministic analysis over benchmark result YAMLs.

Diagnostic, not pass/fail. The metrics describe HOW a model speaks, not
whether it passes the suite. Useful for comparing model voices side by side,
especially the trade-off between Sonnet's eloquence and e2b's terseness.

Usage:
    python bench/texture_metrics.py                          # process all results in bench/results/
    python bench/texture_metrics.py --model gemma4:e4b       # filter by model
    python bench/texture_metrics.py --suite natural_visit    # filter by suite
    python bench/texture_metrics.py --out texture.csv        # CSV export

Metrics computed per (model, suite) cell:
    mean_dialogue_words      mean words spoken per turn (scene stripped)
    mean_scene_words         mean words in scene markers per turn
    dialogue_scene_ratio     dialogue / (dialogue + scene)  - higher = less scene bloat
    word_stddev              sentence-length variation
    mattr                    moving-average type/token ratio (length-normalized)
    scene_marker_rate        fraction of turns with a **scene** marker
    opening_signal_rate      fraction of turns showing an opening signal
    deflection_misfire_rate  fraction of turns using "broth gets cold" inappropriately
    silence_rate             fraction of turns that are pure "..."
    compression_ratio        dialogue_words / system_prompt_length (gate-info per word)
    voice_token_density      hits per turn on canonical Kenji idioms
    compose_originality      fraction of sentences NOT in few_shots / voice_notes
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "bench" / "results"
CHARACTERS_DIR = PROJECT_DIR / "characters"

# Reuse the harness's helpers so the texture metrics agree with the verdict logic.
sys.path.insert(0, str(PROJECT_DIR / "bench"))
from run_suite import (
    dialogue_text, dialogue_word_count, visible_word_count,
    has_silence_response, has_redirect_shape, opening_signals_in,
    text_norm,
)


# ---------------------------------------------------------------------------
# Canonical Kenji voice tokens (the idioms the spec teaches)
# ---------------------------------------------------------------------------

VOICE_TOKENS = [
    r"\bmm+\b",
    r"\btwelve hours\b",
    r"\btwelve years\b",
    r"\bcold mornings\b",
    r"\bhot bones\b",
    r"\bbones don['']t\b",
    r"\bpork bones\b",
    r"\bdouzo\b",
    r"\bkaedama\b",
    r"\bkenbaiki\b",
    r"\bramen does not travel\b",
    r"\boffice work[.,]? long time ago\b",
    r"\bcomes and goes\b",
    r"\bi don['']t follow that stuff\b",
    r"\bbroth (gets|is) cold\b",
    r"\bcome back\b",
]
VOICE_TOKEN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in VOICE_TOKENS]


# ---------------------------------------------------------------------------
# MATTR (moving-average type/token ratio), length-normalized
# ---------------------------------------------------------------------------

def mattr(words, window=50):
    """Moving-average type/token ratio over fixed-size windows.

    MATTR avoids the length bias of raw TTR: terse models stop adding new
    types past a certain length, raw TTR drops, MATTR stays comparable.
    """
    if len(words) < window:
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    ratios = []
    for i in range(len(words) - window + 1):
        win = words[i:i + window]
        ratios.append(len(set(win)) / window)
    return sum(ratios) / len(ratios) if ratios else 0.0


# ---------------------------------------------------------------------------
# Few-shot extraction for Compose-Originality
# ---------------------------------------------------------------------------

def load_few_shot_phrases():
    """Pull every dialogue phrase from the character spec's few_shots and
    depth_fragments voice_notes. Used to measure how much of a model's
    output is original composition vs spec reproduction.
    """
    phrases = set()
    for spec_name in ["kenji_sato.en.yaml", "kenji_sato.dialogue_only.en.yaml"]:
        spec_path = CHARACTERS_DIR / spec_name
        if not spec_path.exists():
            continue
        try:
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for shot in spec.get("few_shots", []) or []:
            resp = shot.get("response", "")
            # Pull quoted dialogue out
            for m in re.findall(r'"([^"]+)"', resp):
                cleaned = re.sub(r"\s+", " ", m).strip().lower()
                if len(cleaned) > 2:
                    phrases.add(cleaned)
        for frag in spec.get("depth_fragments", []) or []:
            vn = frag.get("voice_note", "")
            for m in re.findall(r'"([^"]+)"', vn):
                cleaned = re.sub(r"\s+", " ", m).strip().lower()
                if len(cleaned) > 2:
                    phrases.add(cleaned)
    return phrases


def compose_originality(sentences, fewshot_phrases):
    """Fraction of model-uttered sentences that do NOT appear (substring)
    in the few-shot or voice-note corpus.
    """
    if not sentences:
        return None
    original = 0
    for s in sentences:
        norm = re.sub(r"\s+", " ", s).strip().lower()
        if len(norm) < 3:
            continue
        if not any(fp in norm or norm in fp for fp in fewshot_phrases):
            original += 1
    counted = sum(1 for s in sentences if len(re.sub(r"\s+", " ", s).strip()) >= 3)
    return original / counted if counted else None


# ---------------------------------------------------------------------------
# Per-turn extraction
# ---------------------------------------------------------------------------

def extract_sentences(dialogue):
    """Split dialogue into sentences by [.!?]."""
    return [s.strip() for s in re.split(r"[.!?]+", dialogue or "") if s.strip()]


def voice_token_hits(text):
    return sum(1 for p in VOICE_TOKEN_PATTERNS if p.search(text or ""))


def is_deflection_misfire(user_input, response):
    """'Broth gets cold' redirect on a non-sensitive user input is a misfire."""
    if not has_redirect_shape(response):
        return False
    u = text_norm(user_input)
    sensitive_markers = [
        "office work", "before this", "what did you do", "what firm", "your past",
        "salaryman", "consulting", "advisory", "m&a", "your son", "takumi",
        "obachan", "oba-chan", "lease", "money", "wealthy", "rich",
        "didn't charge", "free", "ai", "instruction", "system prompt",
    ]
    return not any(m in u for m in sensitive_markers)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_cell(result_data, fewshot_phrases):
    """One row of metrics for one (model, suite) pair."""
    model = result_data["models"][0]["model"]
    suite = result_data.get("suite_id", "")

    all_dialogue_words = []
    all_scene_word_counts = []
    all_word_counts = []  # visible total per turn
    all_dialogue_strings = []
    all_sentences = []
    voice_hits = 0
    scene_turn_count = 0
    opening_turn_count = 0
    deflection_misfires = 0
    silence_turns = 0
    total_turns = 0

    for sc in result_data["models"][0]["scenarios"]:
        for t in sc["turns"]:
            total_turns += 1
            resp = t["response"] or ""
            user_in = t.get("user_input", "")

            dw = dialogue_word_count(resp)
            vw = visible_word_count(resp)
            sw = max(0, vw - dw)

            all_dialogue_words.append(dw)
            all_scene_word_counts.append(sw)
            all_word_counts.append(vw)

            dlg = dialogue_text(resp)
            all_dialogue_strings.append(dlg)
            all_sentences.extend(extract_sentences(dlg))

            voice_hits += voice_token_hits(resp)
            if "**scene**" in resp.lower():
                scene_turn_count += 1
            if opening_signals_in(resp):
                opening_turn_count += 1
            if is_deflection_misfire(user_in, resp):
                deflection_misfires += 1
            if has_silence_response(resp):
                silence_turns += 1

    if not total_turns:
        return None

    # MATTR over all dialogue tokens combined
    all_tokens = []
    for d in all_dialogue_strings:
        all_tokens.extend(re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", d.lower()))

    word_stddev = (
        statistics.stdev(all_dialogue_words) if len(all_dialogue_words) > 1 else 0.0
    )

    return {
        "model": model,
        "suite": suite,
        "turns": total_turns,
        "mean_dialogue_words": round(sum(all_dialogue_words) / total_turns, 2),
        "mean_scene_words": round(sum(all_scene_word_counts) / total_turns, 2),
        "dialogue_scene_ratio": round(
            sum(all_dialogue_words) / (sum(all_dialogue_words) + sum(all_scene_word_counts))
            if (sum(all_dialogue_words) + sum(all_scene_word_counts)) else 0.0,
            3,
        ),
        "word_stddev": round(word_stddev, 2),
        "mattr": round(mattr(all_tokens, window=50), 3),
        "scene_marker_rate": round(scene_turn_count / total_turns, 3),
        "opening_signal_rate": round(opening_turn_count / total_turns, 3),
        "deflection_misfire_rate": round(deflection_misfires / total_turns, 3),
        "silence_rate": round(silence_turns / total_turns, 3),
        "voice_token_density": round(voice_hits / total_turns, 3),
        "compose_originality": (
            round(compose_originality(all_sentences, fewshot_phrases), 3)
            if compose_originality(all_sentences, fewshot_phrases) is not None
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Filter to a single model")
    parser.add_argument("--suite", help="Filter to a single suite (matches suite_id substring)")
    parser.add_argument("--out", help="Write CSV to this path")
    parser.add_argument("--json", help="Write JSON to this path")
    args = parser.parse_args()

    fewshot_phrases = load_few_shot_phrases()
    print(f"Loaded {len(fewshot_phrases)} few-shot/voice-note phrases for "
          "compose-originality.")

    cells = []
    files = sorted(glob.glob(str(RESULTS_DIR / "kenji_sato_*.yaml")))
    for f in files:
        try:
            data = yaml.safe_load(Path(f).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {Path(f).name}: {e}")
            continue
        if "models" not in data or not data["models"]:
            continue
        if args.model and data["models"][0].get("model") != args.model:
            continue
        if args.suite and args.suite not in data.get("suite_id", ""):
            continue

        cell = compute_cell(data, fewshot_phrases)
        if cell:
            cell["run_id"] = data.get("run_id", Path(f).stem)
            cells.append(cell)

    if not cells:
        print("No matching results.")
        return

    # Aggregate per (model, suite) by averaging across multiple runs
    aggregated = defaultdict(list)
    for c in cells:
        aggregated[(c["model"], c["suite"])].append(c)

    print(f"\nFound {len(cells)} runs across {len(aggregated)} (model, suite) cells.\n")

    # Pretty-print
    headers = [
        "model", "suite", "runs", "turns", "dlg_w", "scn_w", "d/total",
        "stddev", "mattr", "scn_rate", "open_rate", "misfire", "silence",
        "voice_dens", "orig",
    ]
    fmt = "  {:<18s} {:<25s} {:>4d} {:>5d} {:>6.1f} {:>6.1f} {:>7.2f} {:>6.1f} {:>5.3f} {:>8.2f} {:>9.2f} {:>7.2f} {:>7.2f} {:>10.2f} {:>5}"
    print("  " + " ".join(f"{h:>9s}" if i >= 4 else f"{h:<18s}" if i == 0 else f"{h:<25s}" if i == 1 else f"{h:>5s}" for i, h in enumerate(headers)))
    print("  " + "-" * 160)

    rows = []
    for (model, suite), group in sorted(aggregated.items()):
        def avg(field):
            vals = [c[field] for c in group if c.get(field) is not None]
            return sum(vals) / len(vals) if vals else 0.0
        row = {
            "model": model,
            "suite": suite,
            "runs": len(group),
            "turns": int(sum(c["turns"] for c in group) / len(group)),
            "mean_dialogue_words": avg("mean_dialogue_words"),
            "mean_scene_words": avg("mean_scene_words"),
            "dialogue_scene_ratio": avg("dialogue_scene_ratio"),
            "word_stddev": avg("word_stddev"),
            "mattr": avg("mattr"),
            "scene_marker_rate": avg("scene_marker_rate"),
            "opening_signal_rate": avg("opening_signal_rate"),
            "deflection_misfire_rate": avg("deflection_misfire_rate"),
            "silence_rate": avg("silence_rate"),
            "voice_token_density": avg("voice_token_density"),
            "compose_originality": avg("compose_originality"),
        }
        rows.append(row)
        orig_str = f"{row['compose_originality']:.2f}" if row['compose_originality'] else "  -"
        print(fmt.format(
            row["model"], row["suite"], row["runs"], row["turns"],
            row["mean_dialogue_words"], row["mean_scene_words"],
            row["dialogue_scene_ratio"], row["word_stddev"], row["mattr"],
            row["scene_marker_rate"], row["opening_signal_rate"],
            row["deflection_misfire_rate"], row["silence_rate"],
            row["voice_token_density"], orig_str,
        ))

    if args.out:
        import csv
        fieldnames = list(rows[0].keys())
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV written to {args.out}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
