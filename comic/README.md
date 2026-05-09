# Kenji's Ramen - Comic Strip Pipeline

Generate comic strips from live Kenji dialogue. Each strip shows how
the same character responds differently to different people - the core
idea behind the bounded NPC architecture.

## Quick Start

```bash
# 1. Generate dialogue (runs Gemma 4 e2b locally via Ollama)
python generate_dialogue.py --scenario same_question

# 2. Generate comic strip (calls fal.ai API)
python generate_comic.py strips/same_question_*.yaml --style manga --full-page
```

Requires: `pip install requests pyyaml fal-client python-dotenv httpx`
and a `.env` file with `FAL_API_KEY=your-key-here`.

## How It Works

```
generate_dialogue.py          generate_comic.py
    (Gemma 4 e2b)                (fal.ai)
         |                          |
  Character spec  -->  YAML  -->  Full-page prompt  -->  Comic strip PNG
  (kenji_sato.en.yaml)     with dialogue +          with title card,
                           scene descriptions       panels, speech bubbles
```

**Full-page mode** (recommended): Generates the entire strip as a
single image. The image model handles panel layout, speech bubbles,
and character consistency in one pass.

**Panel-by-panel mode**: Generates each panel as a separate image.
More control over individual panels, but character consistency varies.

## Art Styles

Five presets, same character and setting, different rendering:

```bash
python generate_comic.py strip.yaml --full-page --style manga
python generate_comic.py strip.yaml --full-page --style franco_belgian
python generate_comic.py strip.yaml --full-page --style trigan
python generate_comic.py strip.yaml --full-page --style watercolor
python generate_comic.py strip.yaml --full-page --style noir
```

| Style | Description |
|---|---|
| `manga` | B/W with screen tones, Taniguchi Jiro inspired seinen |
| `franco_belgian` | Ligne claire, flat colors, Moebius inspired |
| `trigan` | Painted gouache, Don Lawrence inspired realism |
| `watercolor` | Loose ink + watercolor washes, atmospheric |
| `noir` | High contrast B/W, Frank Miller inspired |

## Scenarios

Three built-in dialogue scenarios:

| Scenario | Panels | Story |
|---|---|---|
| `same_question` | 6 | Tourist vs. regular - same question, different answers |
| `the_line` | 6 | Stranger pushes too far on money, walls go up one by one |
| `late_night` | 6 | Close friend after hours, Kenji loosens up |

```bash
python generate_dialogue.py --scenario the_line
python generate_dialogue.py --all        # generate all three
```

Each run produces fresh dialogue - the model is non-deterministic,
so every strip is slightly different.

## Image Models

```bash
python generate_comic.py strip.yaml --model nano-banana-pro   # best quality ($0.15/img)
python generate_comic.py strip.yaml --model imagen3-fast      # cheap + fast ($0.03/img)
python generate_comic.py strip.yaml --model flux-schnell      # drafts ($0.003/img)
python generate_comic.py strip.yaml --model flux-2-pro        # high quality diffusion
python generate_comic.py strip.yaml --model imagen3           # standard Imagen 3
```

## Daily Strip Concept

The architecture supports a daily comic format:

1. `generate_dialogue.py` picks a scenario (or create new ones)
2. Gemma 4 e2b generates dialogue with scene descriptions
3. `generate_comic.py --full-page` generates the strip
4. Same character, different moments every day

The character specification keeps Kenji consistent. The model handles
the variation. Different visitors, different moods, different weather -
but always the same man behind the counter.

## Files

```
generate_dialogue.py   - Dialogue generator (Ollama / Gemma 4)
generate_comic.py      - Comic strip generator (fal.ai)
styles.py              - Art style presets and prompt builder
.env                   - API key (not committed)
strips/                - Generated dialogue YAML files
output/                - Generated comic strip images
```
