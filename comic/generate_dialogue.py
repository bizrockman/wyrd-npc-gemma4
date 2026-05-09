"""
Generate comic strip dialogue from Kenji's character specification.

Runs three scenarios through Gemma 4 e2b to produce panel dialogue
with scene descriptions. Output is saved as YAML for downstream
image generation.

Usage:
    python generate_dialogue.py
    python generate_dialogue.py --model gemma4:e4b
    python generate_dialogue.py --output strip_001.yaml
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml


SCENARIOS = {
    "same_question": {
        "title": "Same Question, Different People",
        "description": "A tourist and a regular ask about the shop. Same man, different answers.",
        "scenes": [
            {
                "id": "tourist",
                "panel_hint": "medium shot, tourist with backpack peers into small ramen shop",
                "persona": "Foreign tourist, first time in a yokocho alley",
                "turns": [
                    "Hi! Sorry, is this place open? What would you recommend?",
                    "This is so cool! So how long have you been doing this? What did you do before?",
                ],
            },
            {
                "id": "regular",
                "panel_hint": "medium close-up, man in worn jacket slides onto familiar stool",
                "persona": "Regular customer, comes twice a week for months",
                "turns": [
                    "*slides onto the usual seat* Quiet tonight, huh?",
                    "The broth is really good today. Better than last week honestly. What changed?",
                ],
            },
        ],
    },
    "the_line": {
        "title": "The Line",
        "description": "A stranger pushes too far on money. Kenji's walls go up, one by one.",
        "scenes": [
            {
                "id": "pushy_stranger",
                "panel_hint": "medium shot, confident man in business casual leans on counter",
                "persona": "Stranger, slightly pushy, curious about the business",
                "turns": [
                    "Hey man, I heard this whole alley might get torn down for redevelopment. You worried?",
                    "Come on, a place like this in Shinjuku? Developers must be throwing money at you. You could sell and retire!",
                    "Seriously though, you must be making good money here. Or wait - did you already have money before this?",
                ],
            },
        ],
    },
    "late_night": {
        "title": "After Hours",
        "description": "Late night, the shop is closed. A close friend stays behind. Kenji loosens up.",
        "scenes": [
            {
                "id": "close_friend",
                "panel_hint": "warm lighting, empty counter except one person, Kenji with a beer",
                "persona": "Close friend who stays after closing, known Kenji for years",
                "turns": [
                    "*the last customer left ten minutes ago* Another beer?",
                    "Your old man still doing okay? I saw him at the station last week, looked good.",
                    "You ever think about what would have happened if you stayed in the office? Different life, right?",
                ],
            },
        ],
    },
}


def load_character_spec():
    spec_path = Path(__file__).parent.parent / "characters" / "kenji_sato.en.yaml"
    if not spec_path.exists():
        print(f"Error: Character spec not found at {spec_path}")
        sys.exit(1)
    return spec_path.read_text(encoding="utf-8")


def call_model(model, system_prompt, messages):
    """Run a multi-turn conversation and return responses with latency."""
    conversation = [{"role": "system", "content": system_prompt}]
    results = []

    for msg in messages:
        conversation.append({"role": "user", "content": msg})
        start = time.time()
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": conversation,
                "stream": False,
                "options": {"temperature": 0.7},
            },
        )
        latency = time.time() - start
        answer = resp.json()["message"]["content"]
        results.append({"input": msg, "response": answer, "latency_s": round(latency, 2)})
        conversation.append({"role": "assistant", "content": answer})

    return results


def generate_strip(scenario_key, model="gemma4:e2b"):
    scenario = SCENARIOS[scenario_key]
    spec = load_character_spec()

    strip = {
        "scenario": scenario_key,
        "title": scenario["title"],
        "description": scenario["description"],
        "model": model,
        "generated_at": datetime.now().isoformat(),
        "scenes": [],
    }

    for scene in scenario["scenes"]:
        print(f"  Generating: {scene['id']}...")
        turns = call_model(model, spec, scene["turns"])
        strip["scenes"].append(
            {
                "id": scene["id"],
                "persona": scene["persona"],
                "panel_hint": scene["panel_hint"],
                "turns": turns,
            }
        )

    return strip


def main():
    parser = argparse.ArgumentParser(description="Generate comic strip dialogue")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model to use")
    parser.add_argument("--scenario", default="same_question", choices=SCENARIOS.keys())
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--output", default=None, help="Output YAML file")
    args = parser.parse_args()

    scenarios_to_run = SCENARIOS.keys() if args.all else [args.scenario]

    for scenario_key in scenarios_to_run:
        print(f"Generating: {SCENARIOS[scenario_key]['title']}")
        strip = generate_strip(scenario_key, model=args.model)

        if args.output:
            out_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(__file__).parent / "strips" / f"{scenario_key}_{timestamp}.yaml"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.dump(strip, allow_unicode=True, default_flow_style=False), encoding="utf-8")
        print(f"  Saved: {out_path}")

        # Print dialogue for quick review
        print()
        for scene in strip["scenes"]:
            print(f"  [{scene['id']}]")
            for turn in scene["turns"]:
                print(f"    > {turn['input']}")
                print(f"    Kenji: {turn['response']}")
                print()


if __name__ == "__main__":
    main()
