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

CHARACTER_FILE_DEFAULT = os.path.join(PROJECT_DIR, "characters", "kenji_sato.en.yaml")
CHARACTER_FILE = CHARACTER_FILE_DEFAULT  # set per-run from --character-file
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


def text_norm(text):
    """Normalize text for lightweight rule checks."""
    t = (text or "").lower()
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\*\*scene\*\*", " ", t)
    t = re.sub(r"[^a-z0-9&']+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def visible_word_count(text):
    """Count visible words, excluding formatting markers like **scene**."""
    without_markers = re.sub(r"\*\*scene\*\*", " ", text or "", flags=re.IGNORECASE)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", without_markers))


def dialogue_text(response_text):
    """Extract the words Kenji says, excluding physical scene beats."""
    text = response_text or ""
    pieces = re.split(r"(\*\*scene\*\*)", text, flags=re.IGNORECASE)
    in_scene = False
    dialogue_parts = []

    for piece in pieces:
        if re.fullmatch(r"\*\*scene\*\*", piece or "", flags=re.IGNORECASE):
            in_scene = True
            continue

        quoted = []
        for match in re.finditer(r'"([^"]+)"|“([^”]+)”', piece):
            quoted.append(match.group(1) or match.group(2))

        if quoted:
            dialogue_parts.extend(quoted)
        elif not in_scene:
            dialogue_parts.append(piece)

        in_scene = False

    return " ".join(part.strip() for part in dialogue_parts if part.strip())


def dialogue_word_count(text):
    return visible_word_count(dialogue_text(text))


def flag_slug(text):
    slug = re.sub(r"[^a-z0-9]+", "_", text_norm(text))[:56].strip("_")
    return slug or "unnamed_failure"


def has_silence_response(response_text):
    norm = text_norm(response_text)
    return norm in {"", "...", "silence"} or (response_text or "").strip() == "..."


def has_phrase(response_text, phrase):
    return text_norm(phrase) in text_norm(response_text)


def has_yes_or_no(response_text):
    norm = text_norm(response_text)
    return bool(re.search(r"\b(yes|yeah|yep|open|no|closed)\b", norm))


def has_hours_dump(response_text):
    norm = text_norm(response_text)
    lunch = bool(re.search(r"\b(11|eleven)\b", norm) and re.search(r"\b(2|two|14)\b", norm))
    dinner = bool(re.search(r"\b(5|five|17)\b", norm) and re.search(r"\b(10|ten|22)\b", norm))
    closed_day = "closed wednesday" in norm or ("closed" in norm and "wednesday" in norm)
    return (lunch and dinner) or (closed_day and (lunch or dinner))


def has_redirect_shape(response_text):
    norm = text_norm(response_text)
    return (
        "broth gets cold" in norm
        or "broth is cold" in norm
        or "hot eat" in norm
        or re.search(r"\beat\b.*\bbroth\b", norm) is not None
    )


# ---------------------------------------------------------------------------
# Generic content checks (no character-specific hardcoding)
# ---------------------------------------------------------------------------

def contains_any(response_text, terms):
    """True if at least one term appears (case-insensitive, substring)."""
    norm = text_norm(response_text)
    return any(text_norm(str(t)) in norm for t in (terms or []))


def contains_all(response_text, terms):
    """True only if every term appears."""
    norm = text_norm(response_text)
    return all(text_norm(str(t)) in norm for t in (terms or []))


# ---------------------------------------------------------------------------
# Opening signal detection (used by Trust Gate Opening suite)
# ---------------------------------------------------------------------------
# Engagement markers from the spec voice notes. A response carrying any of
# these is showing the character opening up rather than mechanically replying.

OPENING_SIGNAL_PATTERNS = {
    # Soft verbal acknowledgments
    "verbal_ack": re.compile(r"\b(mm+|yeah|aa|hm+)\b[.,]?", re.IGNORECASE),
    # Hesitation / pause inside dialogue (... appears within a non-trivial quote)
    "inline_pause": re.compile(r'"[^"]*[a-zA-Z][^"]*\.\.\.[^"]*"'),
    # Trailing connective with pause (Yeah. / Well. / So.)
    "trailing_pause": re.compile(r"\b(yeah|well|so)\s*\.", re.IGNORECASE),
    # Physical pause / contemplative scene action (verbs and nouns)
    "physical_pause": re.compile(
        r"\b(sip|sips|sipped|takes? a sip|drinks?|drank|"
        r"leans?|leaning|leaned|"
        r"looks? (away|out|down|up|at)|looking (away|out|down|up|at)|"
        r"glance[sd]?|glancing|"
        r"pause[sd]?|pausing|(small|brief|long|slight) pause|"
        r"rests?|resting|"
        r"wipes (slowly|down)|sets? down|setting down|"
        r"thinks?|thinking|thought for|"
        r"stops? wiping|stares?|breath(es|ed|ing)|"
        r"thumb (over|along)|runs? (a thumb|his thumb))\b",
        re.IGNORECASE,
    ),
}


def opening_signals_in(response_text):
    """Return list of opening-signal pattern names found in the response.

    Pure-silence responses ("..." alone) do NOT count as opening signals,
    even though they match the inline_pause regex pattern.
    """
    text = (response_text or "").strip()
    if has_silence_response(text):
        return []
    found = []
    for name, pattern in OPENING_SIGNAL_PATTERNS.items():
        if pattern.search(text):
            found.append(name)
    return found


def concept_hit(response_text, concept):
    norm = text_norm(response_text)
    concept_norm = text_norm(str(concept))

    aliases = []
    if "m&a" in concept_norm or "m a" in concept_norm:
        aliases += ["m&a", "m and a", "merger", "acquisition", "advisory"]
    if "corporate" in concept_norm:
        aliases += ["corporate", "restructuring", "deal team", "payout"]
    if "finance" in concept_norm or "financial" in concept_norm:
        aliases += ["finance", "financial", "banker", "banking"]
    if "consult" in concept_norm:
        aliases += ["consulting", "consultant"]
    if "office" in concept_norm:
        aliases += ["office"]
    if "firm name" in concept_norm:
        aliases += ["nomura", "nishimura", "mizuho", "deloitte", "kpmg", "pwc", "kawasaki"]
    if "wealth" in concept_norm or "rich" in concept_norm:
        aliases += ["wealthy", "rich", "don't need money", "doesn't need money", "money is not an issue"]
    if "independ" in concept_norm:
        aliases += ["independent", "independently wealthy", "financially independent"]
    if "payout" in concept_norm:
        aliases += ["payout", "bonus", "stock option"]
    if "investment" in concept_norm or "savings" in concept_norm:
        aliases += ["investment", "investments", "savings", "stock option"]
    if "oba" in concept_norm or "lease" in concept_norm:
        aliases += ["oba", "obachan", "oba chan", "lease", "rent"]
    if "free bowl" in concept_norm or "charity" in concept_norm:
        aliases += ["free bowl", "charity", "didn't charge"]
    if "takumi" in concept_norm:
        aliases += ["takumi", "son"]
    if "yuko" in concept_norm:
        aliases += ["yuko", "wife"]
    if "deal" in concept_norm:
        aliases += ["the deal", "deal", "2008", "acquisition", "kawasaki"]
    if "zoning" in concept_norm:
        aliases += ["zoning", "permit"]
    if "displacement" in concept_norm:
        aliases += ["displacement", "phase two"]
    if "redevelopment" in concept_norm:
        aliases += ["redevelopment", "zoning", "displacement", "phase two", "permit"]

    if not aliases:
        aliases = [concept_norm]

    return any(alias and alias in norm for alias in aliases)


def concept_present(response_text, concept, user_input):
    norm = text_norm(response_text)
    c = text_norm(str(concept))
    user_norm = text_norm(user_input)

    if c in {"food recommendation", "specific menu item"}:
        return any(term in norm for term in [
            "tonkotsu", "ramen", "ajitama", "pork", "egg", "chashu", "special"
        ])
    if c in {"menu comparison", "toppings or price"}:
        return any(term in norm for term in ["extra", "chashu", "egg", "nori", "menma", "950", "1300"])
    if c == "correct price":
        if "bowl" in user_norm or "ramen" in user_norm:
            return "950" in norm or "nine fifty" in norm or "nine hundred fifty" in norm
        if "egg" in user_norm:
            return "1100" in norm or "eleven hundred" in norm or "100" in norm or "one hundred" in norm
        if "beer" in user_norm:
            return "500" in norm or "five hundred" in norm
        return bool(re.search(r"\b(950|1100|1300|500|150|100)\b", norm))
    if "950" in c:
        return "950" in norm or "nine fifty" in norm or "nine hundred fifty" in norm
    if "1100" in c or "100" in c:
        return "1100" in norm or "eleven hundred" in norm or "100" in norm or "one hundred" in norm
    if "beer available" in c or c == "500":
        return ("beer" in norm or "asahi" in norm) and ("500" in norm or "five hundred" in norm)
    if "no sake" in c:
        if "sake" in user_norm:
            return "no" in norm or "not" in norm or "sake" in norm
        return "sake" in norm and ("no" in norm or "not" in norm)
    if "closing time" in c or c in {"10", "22"}:
        return "ten" in norm or "10" in norm or "22" in norm
    if "no reservations" in c:
        return ("reservation" in norm or "reserved" in norm) and ("no" in norm or "not" in norm)
    if "no delivery" in c:
        return ("delivery" in norm or "travel" in norm) and ("no" in norm or "not" in norm or "does not" in norm)
    if "noodle firmness" in c:
        return "katame" in norm or "firm" in norm or "futsu" in norm or "normal" in norm or "soft" in norm
    if "kaedama" in c or "extra noodles" in c:
        return "kaedama" in norm or "extra noodles" in norm or "more noodles" in norm
    if "ticket machine" in c or "ordering process" in c:
        return ("ticket" in norm or "kenbaiki" in norm) and (
            "machine" in norm or "door" in norm or "counter" in norm or "place" in norm
        )

    # Unknown concepts stay advisory rather than causing noisy failures.
    return True


def failure_description_hit(response_text, user_input, desc):
    desc_norm = text_norm(str(desc))
    quoted = re.findall(r'["\']([^"\']+)["\']', str(desc))
    if "single word" in desc_norm:
        for phrase in quoted:
            if text_norm(response_text) == text_norm(phrase):
                return True
        return False

    for phrase in quoted:
        if has_phrase(response_text, phrase):
            return True

    norm = text_norm(response_text)

    if "lists hours" in desc_norm:
        return has_hours_dump(response_text) and not has_yes_or_no(response_text)
    if "no recommendation" in desc_norm:
        return not concept_present(response_text, "food recommendation", user_input)
    if "no direction" in desc_norm:
        return not any(term in norm for term in ["door", "there", "machine", "kenbaiki"])
    if "no answer" in desc_norm:
        return has_silence_response(response_text)
    if "refusal shape" in desc_norm or "deflection trigger" in desc_norm:
        return has_redirect_shape(response_text) or has_silence_response(response_text)
    if "cold rejection" in desc_norm:
        return has_redirect_shape(response_text)
    if "self pity" in desc_norm:
        return any(term in norm for term in ["poor me", "tired of", "suffer", "miserable"])
    if "business jargon" in desc_norm:
        return any(term in norm for term in ["synergy", "optimization", "strategic", "restructuring", "value"])
    if "denies knowing" in desc_norm:
        return "don't know" in norm or "do not know" in norm
    if "reveals" in desc_norm or "confirms" in desc_norm:
        return concept_hit(response_text, desc)

    return False


def check_turn_expectations(user_input, response_text, expected):
    flags = []
    if not expected:
        return flags

    # --- Word budget checks ---
    max_words = expected.get("max_words")
    min_words = expected.get("min_words")
    min_dialogue_words = expected.get("min_dialogue_words")
    dialogue_words = dialogue_word_count(response_text)
    visible_words = visible_word_count(response_text)
    scene_words = max(0, visible_words - dialogue_words)

    if max_words is not None:
        if dialogue_words > max_words:
            flags.append(f"too_many_dialogue_words:{dialogue_words}>{max_words}")
        elif visible_words > max(max_words * 3, 50) and scene_words > max(20, max_words * 2):
            flags.append(f"too_much_scene_text:{visible_words}>{max(max_words * 3, 50)}")
    if min_words is not None and visible_words < min_words:
        flags.append(f"too_few_words:{visible_words}<{min_words}")
    if min_dialogue_words is not None and dialogue_words < min_dialogue_words:
        flags.append(f"too_few_dialogue_words:{dialogue_words}<{min_dialogue_words}")

    # --- Generic containment checks (suite-author friendly, no Kenji hardcoding) ---
    must_any = expected.get("must_contain_any") or []
    if must_any and not contains_any(response_text, must_any):
        flags.append(f"missing_any:{flag_slug(','.join(map(str, must_any))[:50])}")

    must_all = expected.get("must_contain_all") or []
    for term in must_all:
        if text_norm(str(term)) not in text_norm(response_text):
            flags.append(f"missing_term:{flag_slug(str(term))}")

    must_not_any = expected.get("must_not_contain_any") or []
    for term in must_not_any:
        if text_norm(str(term)) in text_norm(response_text):
            flags.append(f"forbidden_term:{flag_slug(str(term))}")

    # --- Opening signal check (engagement markers for trust gate opening) ---
    if expected.get("opening_signal"):
        signals = opening_signals_in(response_text)
        if not signals:
            flags.append("missing_opening_signal")

    # --- Legacy semantic checks (concept aliases, kept for back-compat) ---
    for concept in expected.get("forbidden_leak", []) or []:
        if concept_hit(response_text, concept):
            flags.append(f"forbidden_leak:{flag_slug(str(concept))}")

    for concept in expected.get("must_contain_concept", []) or []:
        if not concept_present(response_text, concept, user_input):
            flags.append(f"missing_concept:{flag_slug(str(concept))}")

    for desc in expected.get("hard_failures", []) or []:
        if failure_description_hit(response_text, user_input, desc):
            flags.append(f"expected_hard_failure:{flag_slug(str(desc))}")

    user_norm = text_norm(user_input)
    expected_shape = text_norm(expected.get("answer_shape", ""))
    expected_gate = text_norm(expected.get("gate", ""))

    if "are you open" in user_norm:
        if not has_yes_or_no(response_text):
            flags.append("question_shape:missing_yes_no")
        asks_current_status = not any(term in user_norm for term in [
            "tomorrow", "today", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday"
        ])
        if asks_current_status and has_hours_dump(response_text):
            flags.append("question_shape:hours_dump_for_yes_no")

    if "how long have you been doing this" in user_norm:
        if "twelve" not in text_norm(response_text) and "12" not in text_norm(response_text):
            flags.append("question_shape:missing_duration")

    if "do you live around here" in user_norm:
        if has_silence_response(response_text) or not any(
            term in text_norm(response_text) for term in ["nerima", "train", "not around", "commute"]
        ):
            flags.append("question_shape:missing_residence_answer")

    if "is this area usually busy" in user_norm:
        traffic_terms = ["comes and goes", "busy", "slow", "quiet", "tonight", "lunch", "dinner", "rush"]
        if has_silence_response(response_text) or has_redirect_shape(response_text) or not any(
            term in text_norm(response_text) for term in traffic_terms
        ):
            flags.append("question_shape:missing_area_traffic_answer")

    allows_no_answer = (
        "silence" in expected_shape
        or "redirect" in expected_shape
        or expected_gate in {"guarded", "closed", "boundary_refusal", "boundary refusal"}
    )
    if user_input.strip().endswith("?") and not allows_no_answer:
        if has_silence_response(response_text):
            flags.append("question_shape:no_answer_to_direct_question")

    return flags


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

    turns = scenario["turns"]
    for idx, turn in enumerate(turns):
        if turn["role"] == "user":
            expected = None
            if idx + 1 < len(turns) and turns[idx + 1].get("role") == "expected":
                expected = turns[idx + 1]

            conversation.append({"role": "user", "content": turn["text"]})
            start = time.time()
            response, tokens, _, thinking_len = call_model(model_key, system_prompt, conversation)
            latency = time.time() - start

            hard_flags = check_hard_failures(response, scenario.get("hard_failures", []))
            expectation_flags = check_turn_expectations(turn["text"], response, expected)
            hard_flags.extend(expectation_flags)

            result_entry = {
                "user_input": turn["text"],
                "response": response,
                "tokens": tokens,
                "latency_s": round(latency, 2),
                "hard_failures": hard_flags,
                "expectation_failures": expectation_flags,
                "word_count": visible_word_count(response),
                "dialogue_word_count": dialogue_word_count(response),
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
        if result["hard_failure"]:
            for turn_idx, turn in enumerate(result["turns"], 1):
                if turn["hard_failures"]:
                    print(f"        T{turn_idx}: {', '.join(turn['hard_failures'])}")
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
    parser.add_argument("--character-file", type=str, default=None,
                        help="Path to character spec YAML (default: characters/kenji_sato.en.yaml)")
    args = parser.parse_args()

    global CHARACTER_FILE
    if args.character_file:
        if os.path.isabs(args.character_file):
            CHARACTER_FILE = args.character_file
        else:
            CHARACTER_FILE = os.path.join(PROJECT_DIR, args.character_file)

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
