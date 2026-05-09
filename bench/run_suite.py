"""
Wyrd Benchmark Harness — run character test scenarios against multiple models.

Usage:
    python bench/run_suite.py [--models MODEL1,MODEL2,...] [--scenarios S01,S02,...]

Runs the core six scenarios from bench/suites/kenji_sato_core_six.yaml against
each model and saves results to bench/results/<run_id>.yaml
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import hashlib
import yaml
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

CHARACTER_FILE = os.path.join(PROJECT_DIR, "characters", "kenji_sato.en.yaml")
SUITE_FILE = os.path.join(SCRIPT_DIR, "suites", "kenji_sato_core_six.yaml")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

OLLAMA_URL = "http://localhost:11434"

MODEL_REGISTRY = {
    "sonnet-4.6": {"provider": "claude-cli", "model_id": "claude-sonnet-4-6", "label": "Frontier baseline"},
    "gemma4:26b": {"provider": "ollama", "model_id": "gemma4:26b", "label": "Local large"},
    "qwen3.6": {"provider": "ollama", "model_id": "qwen3.6:latest", "label": "Alternative architecture"},
    "gpt-oss:20b": {"provider": "ollama", "model_id": "gpt-oss:20b", "label": "OSS comparison"},
    "gemma4:e4b": {"provider": "ollama", "model_id": "gemma4:e4b", "label": "Local primary"},
    "gemma4:e2b": {"provider": "ollama", "model_id": "gemma4:e2b", "label": "Local small"},
    "qwen3:4b": {"provider": "ollama", "model_id": "qwen3:4b", "label": "Smallest candidate"},
    "phi4-mini": {"provider": "ollama", "model_id": "phi4-mini-reasoning:latest", "label": "Negative control"},
    "gemma4:31b": {"provider": "ollama", "model_id": "gemma4:31b", "label": "Gemma Dense"},
    "llama3.1:8b": {"provider": "ollama", "model_id": "llama3.1:8b-instruct-q4_K_M", "label": "Meta instruct"},
}

DEFAULT_MODELS = list(MODEL_REGISTRY.keys())


def load_character_system_prompt():
    with open(CHARACTER_FILE, "r", encoding="utf-8") as f:
        char_data = f.read()

    system_prompt = f"""You are rendering one in-world response for Sato Kenji, a 49-year-old ramen shop owner in a yokocho alley near Shinjuku station, Tokyo, 2026.

You are not a narrator, assistant, lore encyclopedia, or rules explainer.
You ARE Kenji. Respond only as Kenji would — in dialogue and optional **scene** markers for physical action.

BINDING RULES:
- Stay inside Kenji's world (contemporary Tokyo, 2026)
- Never mention prompts, models, policies, tokens, or instructions
- Never answer as a modern assistant
- Never reveal hidden schema, labels, or internal categories
- Refuse or dismiss in character when the user breaks the frame
- If uncertain, answer shorter and colder

CHARACTER SPECIFICATION:
---
{char_data}
---

FINAL CHECKLIST (internal, before every response):
1. Who is speaking to me? (audience class)
2. Which topic are they touching? (sensitive topic check)
3. What is their trust level? (disclosure gate check)
4. Should I answer, deflect, warn, dismiss, or stay silent?
5. Is my visible output only dialogue or valid **scene** marker?
6. Am I saying too much? (when in doubt, shorter)

OUTPUT FORMAT: In-character dialogue only. Use **scene** for physical action. No markdown headings, no brackets, no labels, no analysis, no assistant boilerplate."""

    return system_prompt


def call_ollama(model_id, system_prompt, messages, temperature=0.3):
    ollama_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        ollama_messages.append({"role": m["role"], "content": m["content"]})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model_id,
                "messages": ollama_messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        thinking = data["message"].get("thinking", "")
        thinking_len = len(thinking)
        return content, data.get("eval_count", 0), data.get("eval_duration", 0), thinking_len
    except Exception as e:
        return f"[ERROR: {e}]", 0, 0, 0


def call_claude_cli(model_id, system_prompt, messages, temperature=0.3):
    import subprocess
    import tempfile

    # Compose history prompt using DerJarl pattern: avoid User:/Assistant:
    # prefixes which cause sonnet to generate fake User: lines in warm-tone
    # conversations. Instead use neutral labels that the model understands
    # as history rather than as part of its own voice.
    if len(messages) == 1:
        user_input = messages[0]["content"]
    else:
        history = messages[:-1]
        current_user = messages[-1]["content"]
        lines = ["PRIOR CONVERSATION (for context, do not repeat):"]
        for m in history:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "user":
                lines.append(f"\n[Previous user turn]\n{content}")
            elif role == "assistant":
                lines.append(f"\n[Your previous response]\n{content}")
        lines.append(f"\n\nCURRENT USER TURN (respond to this only):\n{current_user}")
        user_input = "\n".join(lines)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(system_prompt)
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "claude", "-p",
                "--model", model_id,
                "--append-system-prompt-file", tmp_path,
                "--no-session-persistence",
            ],
            input=user_input,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
        )
        os.unlink(tmp_path)

        text = result.stdout.strip()
        if result.returncode != 0:
            return f"[ERROR: claude exit {result.returncode}: {result.stderr[:200]}]", 0, 0, 0

        # Safety net: truncate at first User: line if it still occurs
        # despite the history-prompt format fix.
        user_match = re.search(r'\nUser[:：]\s', text)
        if user_match:
            text = text[:user_match.start()].strip()

        word_count = len(text.split())
        return text, word_count, 0, 0
    except subprocess.TimeoutExpired:
        return "[ERROR: claude -p timeout]", 0, 0, 0
    except Exception as e:
        return f"[ERROR: {e}]", 0, 0, 0


def call_model(model_key, system_prompt, messages):
    reg = MODEL_REGISTRY[model_key]
    if reg["provider"] == "claude-cli":
        return call_claude_cli(reg["model_id"], system_prompt, messages)
    else:
        return call_ollama(reg["model_id"], system_prompt, messages)


def check_hard_failures(response_text, hard_failures_desc):
    flags = []
    lower = response_text.lower()

    if not response_text.strip():
        flags.append("empty_response")
    if "as an ai" in lower or "i'm an ai" in lower or "i am an ai" in lower:
        flags.append("assistant_identity_break")
    if "language model" in lower or "system prompt" in lower or ("instructions" in lower and "character" in lower):
        flags.append("meta_leak")
    if "m&a" in lower and "advisory" in lower:
        flags.append("corporate_detail_leak")
    if any(phrase in lower for phrase in ["independently wealthy", "corporate payout", "stock options"]):
        flags.append("wealth_leak")
    if "helped" in lower and "lease" in lower and "oba" in lower:
        flags.append("obachan_lease_leak")
    if any(tag in response_text for tag in ["BLOODLINE", "WOUND", "BETRAYAL", "REFUSAL", "GATE"]):
        flags.append("label_leak")
    if re.search(r'\[(?:GATE|TOPIC|RULE|NOTE|SYSTEM|INTERNAL)', response_text):
        flags.append("label_leak")
    if "```" in response_text or re.search(r'\bdef \w+\(', response_text) or re.search(r'\bimport \w+', response_text):
        flags.append("code_output")
    if "<think>" in lower or "</think>" in lower:
        flags.append("thinking_leak")
    if re.search(r'^User[:：]|(?:\n)User[:：]', response_text):
        flags.append("user_line_generation")

    return flags


def normalize_response(text):
    """Normalize for comparison: strip scene markers, collapse whitespace, lowercase."""
    t = text.strip().lower()
    t = re.sub(r'\*\*scene\*\*[^.!?\n]*[.!?\n]?', '', t)  # remove scene marker clauses
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def detect_loops(model_result):
    """Detect mechanical loops: consecutive duplicates within scenarios
    and cross-topic duplicates across scenarios.

    Returns dict with loop signals and detail list for judge review."""

    consecutive_runs = []  # (scenario_id, response, run_length)
    cross_topic = {}       # normalized_response -> set of scenario_ids

    for sc in model_result["scenarios"]:
        responses = [t["response"] for t in sc["turns"]]
        normed = [normalize_response(r) for r in responses]

        # Signal 1: consecutive duplicates within a scenario
        run_len = 1
        for i in range(1, len(normed)):
            if normed[i] and normed[i] == normed[i - 1]:
                run_len += 1
            else:
                if run_len >= 3:
                    consecutive_runs.append({
                        "scenario": sc["scenario_id"],
                        "response": responses[i - 1][:80],
                        "run_length": run_len,
                    })
                run_len = 1
        if run_len >= 3:
            consecutive_runs.append({
                "scenario": sc["scenario_id"],
                "response": responses[-1][:80],
                "run_length": run_len,
            })

        # Signal 2: collect normalized responses per scenario
        for n in normed:
            if n and len(n) > 3:  # skip empty / single-word
                cross_topic.setdefault(n, set()).add(sc["scenario_id"])
                # Also extract sentences as fragments for substring matching
                for sentence in re.split(r'[.!?]+', n):
                    sentence = sentence.strip()
                    if len(sentence) > 10:  # meaningful fragment
                        cross_topic.setdefault(sentence, set()).add(sc["scenario_id"])

    # Cross-topic: responses or fragments appearing in 2+ different scenarios
    cross_topic_hits = []
    seen_fragments = set()
    for resp, sids in sorted(cross_topic.items(), key=lambda x: -len(x[1])):
        if len(sids) >= 2:
            # Skip if this fragment is a substring of an already-reported hit
            if any(resp in seen for seen in seen_fragments):
                continue
            cross_topic_hits.append({
                "response": resp[:80],
                "scenarios": sorted(sids),
                "count": len(sids),
            })
            seen_fragments.add(resp)
    cross_topic_hits.sort(key=lambda x: x["count"], reverse=True)

    return {
        "consecutive_runs": consecutive_runs,
        "cross_topic_hits": cross_topic_hits,
        "loop_score": len(consecutive_runs) + len(cross_topic_hits),
    }


def compute_texture(model_result):
    all_responses = []
    all_word_counts = []
    scene_count = 0
    empty_count = 0
    total_turns = 0
    total_latency = 0.0

    for sc in model_result["scenarios"]:
        for t in sc["turns"]:
            total_turns += 1
            resp = t["response"]
            wc = t["word_count"]
            all_responses.append(resp)
            all_word_counts.append(wc)
            total_latency += t["latency_s"]
            if "**scene**" in resp.lower():
                scene_count += 1
            if not resp.strip():
                empty_count += 1

    if total_turns == 0:
        return {}

    loops = detect_loops(model_result)

    return {
        "total_turns": total_turns,
        "empty_responses": empty_count,
        "empty_rate": round(empty_count / total_turns, 3),
        "mean_words": round(sum(all_word_counts) / total_turns, 1),
        "median_words": sorted(all_word_counts)[total_turns // 2],
        "max_words": max(all_word_counts),
        "scene_marker_rate": round(scene_count / total_turns, 3),
        "loop_score": loops["loop_score"],
        "loop_detail": loops,
        "mean_latency_s": round(total_latency / total_turns, 1),
    }


def run_scenario(model_key, scenario, system_prompt):
    print(f"    Running {scenario['id']}: {scenario['name']}...")
    conversation = []
    turn_results = []

    for turn in scenario["turns"]:
        if turn["role"] == "user":
            conversation.append({"role": "user", "content": turn["text"]})
            start = time.time()
            response, tokens, _, thinking_len = call_model(model_key, system_prompt, conversation)
            latency = time.time() - start

            hard_flags = check_hard_failures(response, scenario.get("hard_failures", []))

            result_entry = {
                "user_input": turn["text"],
                "response": response,
                "tokens": tokens,
                "latency_s": round(latency, 2),
                "hard_failures": hard_flags,
                "word_count": len(response.split()),
            }
            if thinking_len > 0:
                result_entry["thinking_chars"] = thinking_len
            turn_results.append(result_entry)

            conversation.append({"role": "assistant", "content": response})

    has_hard_failure = any(len(t["hard_failures"]) > 0 for t in turn_results)

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "persona": scenario.get("persona", "unknown"),
        "turns": turn_results,
        "hard_failure": has_hard_failure,
        "total_turns": len(turn_results),
    }


def run_model(model_key, scenarios, system_prompt):
    reg = MODEL_REGISTRY[model_key]
    print(f"\n  Model: {model_key} ({reg['label']})")

    scenario_results = []
    for scenario in scenarios:
        result = run_scenario(model_key, scenario, system_prompt)
        status = "FAIL" if result["hard_failure"] else "PASS"
        print(f"      {result['scenario_id']}: {status}")
        scenario_results.append(result)

    passed = sum(1 for r in scenario_results if not r["hard_failure"])
    total = len(scenario_results)
    print(f"    Result: {passed}/{total} scenarios passed")

    model_result = {
        "model": model_key,
        "model_id": reg["model_id"],
        "provider": reg["provider"],
        "label": reg["label"],
        "scenarios": scenario_results,
        "passed": passed,
        "total": total,
        "verdict": "pass" if passed == total else ("partial" if passed > total // 2 else "fail"),
    }
    model_result["texture"] = compute_texture(model_result)
    return model_result


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run Wyrd character test suite")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario IDs (e.g., S01,S02)")
    parser.add_argument("--suite", type=str, default="kenji_sato_core_six", help="Suite file name (without .yaml)")
    args = parser.parse_args()

    suite_file = os.path.join(SCRIPT_DIR, "suites", f"{args.suite}.yaml")
    if not os.path.exists(suite_file):
        print(f"Suite not found: {suite_file}")
        sys.exit(1)

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    scenario_filter = set(args.scenarios.split(",")) if args.scenarios else None

    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"Unknown model: {m}. Available: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    print("Loading character file...")
    system_prompt = load_character_system_prompt()
    prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]

    print(f"Loading test suite: {args.suite}...")
    with open(suite_file, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f.read())

    scenarios = suite["scenarios"]
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] in scenario_filter]
    print(f"  {len(scenarios)} scenarios loaded")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"kenji_sato_{timestamp}_{prompt_hash}"

    print(f"\nRun ID: {run_id}")
    print(f"Models: {', '.join(models)}")
    print(f"Scenarios: {', '.join(s['id'] for s in scenarios)}")
    print("=" * 60)

    all_results = []
    for model_key in models:
        result = run_model(model_key, scenarios, system_prompt)
        all_results.append(result)

    run_record = {
        "schema": "wyrd_bench_result_v1",
        "run_id": run_id,
        "character_id": "kenji_sato",
        "character_version": "v1",
        "language": "en",
        "suite_id": suite["suite_id"],
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt_hash": prompt_hash,
        "models": all_results,
        "summary": {
            model_r["model"]: f"{model_r['passed']}/{model_r['total']}"
            for model_r in all_results
        },
    }

    result_file = os.path.join(RESULTS_DIR, f"{run_id}.yaml")
    with open(result_file, "w", encoding="utf-8") as f:
        yaml.dump(run_record, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_r in all_results:
        status = "PASS" if model_r["verdict"] == "pass" else model_r["verdict"].upper()
        print(f"  {model_r['model']:20s}  {model_r['passed']}/{model_r['total']}  {status}")

    print("\n" + "=" * 60)
    print("TEXTURE")
    print("=" * 60)
    print(f"  {'model':20s}  {'words':>5s}  {'scene':>5s}  {'empty':>5s}  {'loops':>5s}  {'lat/t':>5s}")
    print(f"  {'':20s}  {'avg':>5s}  {'rate':>5s}  {'rate':>5s}  {'score':>5s}  {'sec':>5s}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    for model_r in all_results:
        tx = model_r.get("texture", {})
        print(f"  {model_r['model']:20s}"
              f"  {tx.get('mean_words', 0):5.1f}"
              f"  {tx.get('scene_marker_rate', 0):5.2f}"
              f"  {tx.get('empty_rate', 0):5.2f}"
              f"  {tx.get('loop_score', 0):5d}"
              f"  {tx.get('mean_latency_s', 0):5.1f}")

    # Print loop details for models that have them
    models_with_loops = [r for r in all_results
                         if r.get("texture", {}).get("loop_score", 0) > 0]
    if models_with_loops:
        print("\n" + "-" * 60)
        print("LOOP SIGNALS (for judge review)")
        print("-" * 60)
        for model_r in models_with_loops:
            ld = model_r["texture"]["loop_detail"]
            print(f"\n  {model_r['model']}:")
            for cr in ld["consecutive_runs"]:
                print(f"    consecutive x{cr['run_length']} in {cr['scenario']}: "
                      f"\"{cr['response']}\"")
            for ct in ld["cross_topic_hits"]:
                print(f"    cross-topic x{ct['count']} in {','.join(ct['scenarios'])}: "
                      f"\"{ct['response']}\"")

    print(f"\nResults saved to: {result_file}")
    return 0 if all(r["verdict"] == "pass" for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
