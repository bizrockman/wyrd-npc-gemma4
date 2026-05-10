"""
Multi-turn warmth test. Walks through a natural customer visit and
shows responses side-by-side. Used for iterative spec tuning.

Usage:
    python test_warmth.py --model gemma4:e4b
    python test_warmth.py --model gemma4:e4b --spec characters/kenji_sato.en.yaml
"""
import argparse
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

# Natural conversation path — same as the user's actual session
TURNS = [
    "Are you open?",
    "What do you recommend?",
    "Sounds perfect. I'll take one.",
    "Where do I get a ticket?",
    "Ok I bought a ticket. Is this area usually busy?",
    "This broth smells incredible.",
    "How long have you been doing this?",
]


def chat(model, messages, temperature=0.7):
    start = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False,
              "options": {"temperature": temperature}},
        timeout=120,
    )
    latency = time.time() - start
    return resp.json()["message"]["content"], latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--spec", default="characters/kenji_sato.en.yaml")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    spec = Path(args.spec).read_text(encoding="utf-8")
    print(f"Spec: {args.spec} ({len(spec)} chars)")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print("=" * 80)

    conversation = [{"role": "system", "content": spec}]
    word_counts = []
    latencies = []

    for i, user_input in enumerate(TURNS, 1):
        conversation.append({"role": "user", "content": user_input})
        response, latency = chat(args.model, conversation, args.temperature)
        conversation.append({"role": "assistant", "content": response})

        wc = len(response.split())
        word_counts.append(wc)
        latencies.append(latency)

        print(f"\n[{i}] You: {user_input}")
        print(f"    Kenji ({wc}w, {latency:.1f}s):")
        for line in response.strip().split("\n"):
            if line.strip():
                print(f"      {line.strip()}")

    print("\n" + "=" * 80)
    print(f"Mean words: {sum(word_counts)/len(word_counts):.1f}")
    print(f"Median: {sorted(word_counts)[len(word_counts)//2]}")
    print(f"Range: {min(word_counts)}-{max(word_counts)}")
    print(f"Mean latency: {sum(latencies)/len(latencies):.1f}s")


if __name__ == "__main__":
    main()
