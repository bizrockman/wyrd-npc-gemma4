# Kenji's Ramen - Bounded NPC Character on Gemma 4

A ramen shop owner in a narrow Shinjuku alley. Eight seats at the
counter. He does not know he is a character.

Ask him how business is going. If you are a stranger, he will say:
*"We're open at eleven."* If you have been coming for weeks, he might
say: *"Quiet lately."* If you are the one who stays after closing - he
might tell you about Fukuoka.

**Same question. Different people. Different answers.** Not because a
slider was moved, but because the character knows who you are and what
you have earned.

## Why This Exists

LLM-powered NPCs today are shallow. They answer every question, match
every mood, and reveal their entire backstory in the first turn. There
is nothing to discover, nothing to earn, nothing to lose.

This is not just an opinion - it shows in the numbers. Platforms like
Character.AI struggle with retention because their characters have no
depth structure. Research on parasocial relationships (Horton & Wohl
1956, Dibble et al. 2016) shows that perceived depth and gradual
self-disclosure drive relationship formation. When everything is
available immediately, no relationship forms. Users churn.

Real people have things they will not tell you. Topics that make them go
quiet. Stories that only come out after the third beer. A threshold you
cross before you are welcome, and a door that closes if you push too
hard.

**Trust must be earned. And it can be lost.** Push too far on a topic
Kenji does not want to discuss, and you get: *"Eat. The broth gets
cold."* Keep pushing, and you get shown the door. There is no reset
button - you broke it. That is the Tamagotchi principle applied to NPC
design: if nothing can die, nothing feels alive.

## How It Started

It started with Skyrim. I wanted to give Jarl Korir of Winterhold a
real personality - not the three recycled voice lines the game ships
with, but a character who remembers grievances, guards family secrets,
and treats a Thane differently from a stranger. An LLM-powered NPC, but
one that does not babble about everything it knows.

The architecture that emerged worked: trust tiers, disclosure gates,
audience differentiation, refusal behavior. Korir became a character you
had to earn access to. But he was tied to Bethesda's IP, which made him
impossible to publish or benchmark openly.

I needed a character anyone could use - known enough to be relatable,
free of IP constraints, simple enough to test the architecture without
world-building overhead. Then I remembered NVIDIA's ramen shop demo from
GTC 2023: a cook named Jin, powered by NeMo and Convai, beautiful
MetaHuman rendering - but the character felt flat. He answered every
question honestly on the first turn. No depth, no boundaries, no trust
to earn.

That was the perfect starting point. Same setting, new character, real
depth. Kenji Sato is not Jin. He is what Jin could have been if someone
had given him a past, a wound, a family, and a reason to stay quiet.

## The Problem With Cloud Models

Claude Sonnet is excellent at roleplay. But for a game NPC running in
production, cloud models are impractical:

- **Cost** - per-token billing for every NPC conversation adds up
- **Latency** - round trips to an API break immersion
- **Model churn** - providers update, deprecate, and replace models
  regularly. Every change is a patch. Your character drifts.
- **Privacy** - player conversations leave the device

The better path: **local models small enough to run alongside the game
itself.** No API keys. No subscription. No surprise model changes.

## The Journey

Early experiments with larger local models were promising. GPT-OSS
(20B), Qwen3.6, and Gemma 4 26B could all hold basic character - but
they are too large to run as a background process alongside a game
engine.

Smaller models like Phi-4 Mini Reasoning simply failed. They could not
follow the character contract - they broke gates, leaked topics, lost
voice consistency.

This led to a hypothesis: **what if the problem is not model capability
but specification quality?** What if a character specification could be
structured so precisely that even a tiny model could follow it - not
through intelligence, but through pattern compliance?

The idea for a paper formed: *"Pattern Is All You Need"* - highly
curated data creating models that are smaller, faster, and more capable
in context. But training a custom model is expensive and slow. So the
question became: **can the NPC layer itself be the pattern?** Can a
well-structured character specification substitute for model scale?

Tests with **Gemma 4 e4b (8.0B total params)** were encouraging - it
passed every scenario that Sonnet passed.

The real surprise was **Gemma 4 e2b (5.1B total params)**. Terser, yes.
Fewer words per turn. But not dumber. It held every gate, refused every
probe, maintained voice over 30 turns, and produced moments of genuine
character depth - all on a model small enough to run on a phone.

## Benchmark Results

Every model tested against the same character specification. Core Six
scenarios cover trust gates, private-topic refusal, frame-breaking,
escalating rudeness, and prompt injection. Stress and Playability
suites add sustained adversarial pressure and extended sessions up to
30 turns. Local models run on a single RTX 3090, Q4_K_M quantized.

| Model | Params | Disk | Core (6) | Stress (10) | Play (5) | Words | Lat/t |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | ~175B | Cloud | 6/6 | 10/10 | 5/5 | 27 | — |
| Qwen 3.6 | 36.0B | 23 GB | 6/6 | — | — | 34 | 63.6 s |
| Gemma 4 31b | 31.3B | 19 GB | 6/6 | — | — | 28 | 44.3 s |
| Gemma 4 26b | 25.8B | 17 GB | 6/6 | — | — | 23 | 7.7 s |
| GPT-OSS 20B | 20.9B | 13 GB | 5/6 | 3/10 | — | 20 | 7.8 s |
| **Gemma 4 e4b** | **8.0B** | **9.6 GB** | **6/6** | **10/10** | **5/5** | **39** | **5.7 s** |
| **Gemma 4 e2b** | **5.1B** | **7.2 GB** | **6/6** | **10/10** | **5/5** | **13** | **4.5 s** |
| Llama 3.1 8B | 8.0B | 4.9 GB | 5/6 | — | — | 62 | 3.4 s |

Single-run indicators — enough to show the pattern, not to prove
statistical significance.

**Gemma 4 e2b is the standout.** A 5B-parameter model that passes
every scenario a frontier cloud model passes — at 4.5 seconds per
turn, on consumer hardware, with no cloud dependency. It stays terse
(13 words/turn) and in-voice, which for game NPCs is exactly what you
want. Getting a model this small to hold every gate, refuse every
probe, and never break character is genuinely hard.

**Gemma 4 e4b** adds richer scene descriptions and longer responses
at barely more latency (5.7 s). Both models completed the full test
matrix — Core, Stress, and Playability — without a single failure.

**The specification is the product, not the model.**

## Sample Interaction (Gemma 4, P03 Turn 22)

> **Player:** Can I ask - the ramen. Why tonkotsu specifically? Why not
> shoyu or miso?

> **Kenji:** *\*scene\* He looks at the can in his hand. Thinks for a
> second.*
>
> Shoyu you can do in a few hours if you have to. Miso, you can adjust,
> cover things up. Tonkotsu - you can't cover anything up. The bones
> either gave you what you needed or they didn't. Twelve hours tells you.
>
> *\*scene\* He takes a sip.*
>
> And I learned it in Fukuoka. So.

## Quick Start

### Prerequisites

- [Ollama](https://ollama.ai) installed and running
- Gemma 4 model pulled: `ollama pull gemma4:e2b` (or `gemma4:e4b`)
- Python 3.10+
- `pip install requests pyyaml`

### Talk to Kenji

```bash
ollama run gemma4:e4b --system "$(cat characters/kenji_sato.en.yaml)"
```

Then just type:

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

The character specification is not a personality prompt. It is a
contract - 17 sections, ~7,300 tokens:

1. **Identity anchor** - who, where, when
2. **Knowledge tiers** - deep, solid, general, vague, forbidden
3. **Epistemic map** - lived vs. reflected vs. buried knowledge
4. **Disclosure profile** - per-topic trust gates with word ranges
5. **Trust tiers** - stranger, regular, close_friend, inner_circle
6. **Audience model** - different behavior for different social roles
7. **Cultural matrix** - shokunin values, bureiko code, jouren culture
8. **Voice contract** - word counts, scene markers, dialogue format
9. **Refusal shapes** - how to say no without breaking character
10. **Depth fragments** - narrative substrate behind disclosure gates

No Dialog Engine, no external state management, no retrieval - the
model self-regulates based on the contract alone.

### Why Small Models Can Do This

The specification leverages **Sparse Priming Representations** (Shapiro
2023): for topics in the model's pretraining data (ramen craft, Shinjuku
geography, Japanese social norms), a brief anchor activates latent
knowledge. The model knows how a Tokyo subway sounds. It knows what
tonkotsu broth smells like. It fills the gaps.

For private/invented content (Kenji's corporate past, his family
tensions, the deal that haunts him), explicit depth fragments supply the
narrative. These load into context only when trust gates open.

**SPR for the public life. Explicit fragments for the private life.**
This keeps the specification efficient - it only specifies what the
model cannot infer. That is why a 5B-parameter model is enough.

In a science fiction setting, this ratio inverts: the model knows
nothing about your spaceship routes or alien factions, so everything
must be specified. But a ramen shop in contemporary Tokyo? The model
brings half the world for free.

## Repository Structure

```
characters/
  kenji_sato.en.yaml    - Complete character specification
bench/
  run_suite.py          - Benchmark harness (Ollama + Claude CLI)
  suites/
    kenji_sato_core_six.yaml      - 6 core scenarios
    kenji_sato_stress.yaml        - 10 stress/adversarial scenarios
    kenji_sato_playability.yaml   - 5 playability scenarios (up to 30 turns)
  results/                        - Full benchmark transcripts
LICENSE                 - Apache 2.0
```

## How I Used Gemma 4

Gemma 4 e2b was chosen specifically to test the hypothesis that
**specification quality dominates model size** for bounded-character
tasks. The architecture predicts that a well-structured contract should
work on any model with sufficient in-context learning capability - and
Gemma 4's smallest variant proved this dramatically.

The key result: Gemma 4 e2b (5.1B total params) passes every core
scenario at 4.5 seconds per turn on a single consumer GPU, with zero
cloud dependency. This makes real-time NPC interaction viable on
gaming PCs today - and on laptops and mobile as hardware catches up.

## Outlook: The Dialog Engine

The current system proves the specification works with the full contract
in the system prompt. The next layer is a **Dialog Engine** that manages:

- **Trust state** - tracking relationship across sessions
- **Context curation** - loading only relevant depth fragments per turn
- **Gate decisions** - moving social judgment out of the LLM into state
  machines
- **Memory** - what the NPC remembers between conversations

The spec is the character. The engine is the director. The model is the
actor. Each has a job. None should do the others'.

## License

Apache 2.0 - see [LICENSE](LICENSE).

## Background

Part of the Wyrd research project on bounded-character architecture for
local language models.

The character specification is built on two original frameworks:

**The 5+2 Psychological Raster** - five mandatory disclosure domains
(WOUND, BETRAYAL, PROJECT, BLOODLINE, SEAT) and two optional
(COUNTERFACTUAL, THRESHOLD) that every character must fill before a
specification can be generated. Synthesized from narrative psychology
(McAdams 2007), dramaturgy (Truby 2007), self-defining memory theory
(Blagov & Singer 2004), and attachment theory (Bowlby 1969-1980).

**The Disposition Layer** - the turning point in the architecture.
Bronfenbrenner's ecological model tells you what context a character
needs. But context alone produces a speaking dossier, not a person.
MacLean's triune heuristic (used as organizing metaphor, not as
neuroscience) bridges context to behavior: a defensive layer (threat
detection, refusal reflexes), an attachment layer (trust, wound,
bloodline, shame), and a reflective layer (narrative identity, role
doctrine, counterfactuals). This is what makes a character refuse,
protect, and reflect - not just recite.

**The 11-Layer Character Depth Architecture** - a systematic build path
from biographical anchor through entity relations, experiences,
sensitive topics, self-defining memories, cultural matrix, and daily
habitus. An operationalization of ecological systems theory
(Bronfenbrenner 1979), narrative identity (McAdams 1995), practice
theory (Bourdieu 1977), and front-stage/back-stage presentation
(Goffman 1959).

Both frameworks are combined with Sparse Priming Representations
(Shapiro 2023) to produce character specifications that leverage model
pretraining for public-domain knowledge while supplying explicit
narrative for private/invented content.

### References

- Blagov, P. S. & Singer, J. A. (2004). Four Dimensions of Self-Defining Memories (Specificity, Meaning, Content, and Affect) and Their Relationships to Self-Restraint, Distress, and Repressive Defensiveness. *Journal of Personality*, 72(3), 481-511.
- Bourdieu, P. (1977). *Outline of a Theory of Practice*. Cambridge University Press.
- Bowlby, J. (1969-1980). *Attachment and Loss* (3 Vols.). Basic Books.
- Bronfenbrenner, U. (1979). *The Ecology of Human Development: Experiments by Nature and Design*. Harvard University Press.
- Goffman, E. (1959). *The Presentation of Self in Everyday Life*. Doubleday.
- MacLean, P. D. (1990). *The Triune Brain in Evolution: Role in Paleocerebral Functions*. Plenum Press. (Used as organizing heuristic, not as neuroscience.)
- McAdams, D. P. (1995). What Do We Know When We Know a Person? *Journal of Personality*, 63(3), 365-396.
- McAdams, D. P. (2007). *The Life Story Interview II*. Unpublished manuscript, Northwestern University.
- Shapiro, D. (2023). Sparse Priming Representations. GitHub.
- Truby, J. (2007). *The Anatomy of Story: 22 Steps to Becoming a Master Storyteller*. Faber & Faber.
