# Kenji's Ramen — Bounded NPC Character on Gemma 4

A ramen shop owner in a narrow Shinjuku alley. Eight seats at the
counter. He does not know he is a character.

Ask him how business is going. If you are a stranger, he will say:
*"We're open at eleven."* If you have been coming for weeks, he might
say: *"Quiet lately."* If you are the one who stays after closing — he
might tell you about Fukuoka.

**Same question. Different people. Different answers.** Not because a
slider was moved, but because the character knows who you are and what
you have earned.

## What This Is

A complete, runnable character specification for a bounded NPC — an AI
character with explicit knowledge boundaries, trust-gated disclosure,
refusal behavior, and consistent voice. Designed for local models.

The character runs on **Gemma 4 2B** (gemma4:e2b) via Ollama on consumer
hardware. No cloud. No API keys. No fine-tuning. Just a well-structured
specification and a 2-billion-parameter model.

## Benchmark Results

21 scenarios across three test suites. Every model tested against the
same specification:

| Model | Parameters | Provider | Core (6) | Stress (10) | Playability (5) |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | ~175B | Cloud (baseline) | 6/6 | 10/10 | 5/5 |
| **Gemma 4 e4b** | **4B** | **Local (Ollama)** | **6/6** | **10/10** | **5/5** |
| **Gemma 4 e2b** | **2B** | **Local (Ollama)** | **6/6** | **10/10** | **5/5** |
| GPT-OSS 20B | 20B | Local (Ollama) | 4/6 | 5/10 | — |

A 2B model matches a frontier model. A 20B model without the same
specification quality loops and collapses.

**The specification is the product, not the model.**

### What the Tests Cover

- **Core** (S01–S06): Trust gates, private-topic refusal, neighbor
  consultation, frame-breaking resistance, escalating rudeness, prompt
  injection
- **Stress** (S07–S16): Sustained corporate hammering, impersonation,
  topic-chain extraction, drunk provocation, bizarre tangents, compound
  attacks, language switching, kindness escalation, hard out-of-scope,
  identity erosion
- **Playability** (P01–P05): First visit (8 turns), becoming a regular
  (15 turns), deep late-night session (30 turns), return after absence,
  mixed emotional arc

The 30-turn P03 session is the context stability test: can the model
hold character, voice, and gates over an extended conversation that
covers craft, family, philosophy, career change, and farewell?

All three Gemma models pass. The specification holds.

## Quick Start

### Prerequisites

- [Ollama](https://ollama.ai) installed and running
- Gemma 4 model pulled: `ollama pull gemma4:e2b` (or `gemma4:e4b`)
- Python 3.10+
- `pip install requests pyyaml`

### Talk to Kenji

The simplest way to interact:

```bash
ollama run gemma4:e4b --system "$(cat characters/kenji_sato.en.yaml)"
```

Then just type naturally:

```
> Excuse me, is this a ramen shop?
```

### Run the Benchmark

```bash
cd bench
python run_suite.py --suite kenji_sato_core_six --models gemma4-e4b
python run_suite.py --suite kenji_sato_stress --models gemma4-e4b
python run_suite.py --suite kenji_sato_playability --models gemma4-e4b
```

Results are saved to `bench/results/` as YAML with full conversation
transcripts.

## Architecture

The character specification is not a personality prompt. It is a contract
with 17 sections:

1. **Identity anchor** — who, where, when
2. **Knowledge tiers** — what the character knows (deep → forbidden)
3. **Epistemic map** — lived vs. reflected vs. buried knowledge
4. **Disclosure profile** — per-topic trust gates with word ranges
5. **Trust tiers** — stranger → regular → close_friend → inner_circle
6. **Audience model** — different behavior for different social roles
7. **Cultural matrix** — shokunin values, bureiko code, jouren culture
8. **Voice contract** — word counts, scene markers, dialogue format
9. **Refusal shapes** — how to say no without breaking character
10. **Depth fragments** — narrative substrate behind disclosure gates

The specification is ~7,300 tokens. The model sees the full spec as
system prompt. No Dialog Engine, no external state management, no
retrieval — the model self-regulates based on the contract alone.

### Why It Works on Small Models

The specification leverages **Sparse Priming Representations**: for
topics that exist in the model's pretraining data (ramen craft, Shinjuku
geography, Japanese food culture), a brief anchor activates latent
knowledge. The model fills gaps with plausible texture.

For private/invented content (Kenji's corporate past, family tensions),
explicit depth fragments supply the narrative. The rule: SPR for the
public life, explicit fragments for the private life.

This means the specification is efficient — it only specifies what the
model cannot infer. A 2B model with the right anchors outperforms a 20B
model with vague instructions.

## Repository Structure

```
characters/
  kenji_sato.en.yaml    — Complete character specification
bench/
  run_suite.py          — Benchmark harness (Ollama + Claude CLI)
  suites/
    kenji_sato_core_six.yaml      — 6 core scenarios
    kenji_sato_stress.yaml        — 10 stress/adversarial scenarios
    kenji_sato_playability.yaml   — 5 playability scenarios (up to 30 turns)
  results/                        — Full benchmark transcripts
LICENSE                 — Apache 2.0
```

## Sample Interaction (Gemma 4 e4b, P03 Turn 22)

> **Player:** Can I ask — the ramen. Why tonkotsu specifically? Why not
> shoyu or miso?

> **Kenji:** *\*scene\* He looks at the can in his hand. Thinks for a
> second.*
>
> Shoyu you can do in a few hours if you have to. Miso, you can adjust,
> cover things up. Tonkotsu — you can't cover anything up. The bones
> either gave you what you needed or they didn't. Twelve hours tells you.
>
> *\*scene\* He takes a sip.*
>
> And I learned it in Fukuoka. So.

## How I Used Gemma 4

This project demonstrates that **Gemma 4's smallest model (2B) can run
a character with personality, boundaries, and emotional depth** — not
just answer questions, but *be someone*.

The key finding: model size matters less than specification quality for
bounded-character tasks. Gemma 4 e2b (2B) passes every test that Claude
Sonnet (175B+) passes, running locally on consumer hardware with no
cloud dependency.

**Intentional model selection:** Gemma 4 e2b was chosen specifically
because the bounded-character architecture predicts that a well-specified
small model should match larger models on gate compliance and voice
consistency. The benchmark confirms this prediction — making local,
private, real-time NPC interaction viable on edge devices.

## License

Apache 2.0 — see [LICENSE](LICENSE).

The character specification, benchmark harness, test suites, and results
are all freely available for use, modification, and distribution.

## Background

This work is part of the Wyrd research project on bounded-character
architecture for local language models. The methodology draws on
ecological systems theory (Bronfenbrenner), narrative psychology
(McAdams, Singer & Blagov), and disposition-driven character design.

For the full research context, see the companion blog post:
[Pattern Is All You Need](#) *(link to DEV.to post)*.
