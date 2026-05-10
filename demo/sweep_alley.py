"""
Sweep test: generate 20 variations of the alley intro with different
prompts, styles, and parameters. Builds a contact sheet at the end
for easy comparison.

Run after .venv-pixelart is set up and SDXL/LoRA cached:
    .venv-pixelart\\Scripts\\activate
    python demo/sweep_alley.py
"""
import io
import math
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

DEMO_DIR = Path(__file__).resolve().parent
OUT_DIR = DEMO_DIR / "assets" / "variations"
PROJECT_DIR = DEMO_DIR.parent

# Common scene description (the alley itself stays the same)
SCENE_BASE = (
    "Japanese yokocho alley at night, two glowing red paper lanterns, "
    "small ramen shop with white noren curtains, wooden counter and stools, "
    "steam rising, brick path"
)

NEGATIVE = (
    "photorealistic, smooth shading, blurry, anti-aliasing, gradients, "
    "modern 3D render, anime cel shading, text, watermark, signature"
)

# 20 variations — style anchors, palettes, moods, perspectives
VARIATIONS = [
    # --- Style anchors (10) ---
    ("01_jrpg_vibrant",   "pixel art, 16-bit JRPG, vibrant colors, "          + SCENE_BASE),
    ("02_nes_8bit",       "pixel art, NES style, 8-bit aesthetic, "           + SCENE_BASE),
    ("03_snes_rpg",       "pixel art, SNES RPG style, "                       + SCENE_BASE),
    ("04_stardew_cozy",   "pixel art, Stardew Valley style, cozy, "           + SCENE_BASE),
    ("05_earthbound",     "pixel art, Earthbound style, surreal colors, "     + SCENE_BASE),
    ("06_gba_clean",      "pixel art, GBA Pokemon style, bright clean, "      + SCENE_BASE),
    ("07_ff6_detailed",   "pixel art, Final Fantasy 6 style, detailed, "      + SCENE_BASE),
    ("08_chronotrigger",  "pixel art, Chrono Trigger style, anime pixel, "    + SCENE_BASE),
    ("09_indie",          "pixel art, retro indie game, hand-painted, "       + SCENE_BASE),
    ("10_neon_synthwave", "pixel art, neon synthwave, vibrant red and blue, " + SCENE_BASE),
    # --- Color & mood (5) ---
    ("11_warm_sunset",    "pixel art, warm sunset colors, golden hour, "      + SCENE_BASE),
    ("12_cyberpunk",      "pixel art, cyberpunk Tokyo, neon-lit alley, "      + SCENE_BASE),
    ("13_autumn",         "pixel art, autumn evening, orange amber, "         + SCENE_BASE),
    ("14_rainy_night",    "pixel art, rainy night, blue cool tones, wet reflections, " + SCENE_BASE),
    ("15_bright_day",     "pixel art, vibrant midday colors, bright daylight, " + SCENE_BASE),
    # --- Composition (5) ---
    ("16_closeup",        "pixel art, close-up shop entrance, detailed lanterns, " + SCENE_BASE),
    ("17_isometric",      "pixel art, isometric view, "                       + SCENE_BASE),
    ("18_sidescroll",     "pixel art, side-scrolling beat-em-up perspective, " + SCENE_BASE),
    ("19_topdown",        "pixel art, top-down JRPG town view, "              + SCENE_BASE),
    ("20_wide_atmos",     "pixel art, wide atmospheric establishing shot, "   + SCENE_BASE),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Loading SDXL + pixel-art-xl LoRA...")

    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.load_lora_weights("nerijs/pixel-art-xl")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    SEED = 42  # fixed seed across variants so the only changing thing is the prompt
    print(f"\nGenerating {len(VARIATIONS)} variants (seed={SEED})...")
    print("=" * 60)

    images = []
    start_total = time.time()
    for i, (name, prompt) in enumerate(VARIATIONS, 1):
        out_path = OUT_DIR / f"{name}.png"
        print(f"  [{i:2d}/{len(VARIATIONS)}] {name}", flush=True)
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        img = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=8,
            guidance_scale=1.5,
            width=1024,
            height=576,
            generator=gen,
        ).images[0]
        img.save(out_path, format="PNG", optimize=True)
        images.append((name, img))

    elapsed = time.time() - start_total
    print(f"\nGeneration done in {elapsed:.1f}s ({elapsed/len(VARIATIONS):.1f}s avg)")

    # Build contact sheet (5x4 grid, each thumbnail at 384x216)
    print("\nBuilding contact sheet...")
    cols, rows = 5, 4
    thumb_w, thumb_h = 384, 216
    label_h = 28
    cell_w, cell_h = thumb_w, thumb_h + label_h
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, (name, img) in enumerate(images):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = row * cell_h
        thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y + thumb_h, x + cell_w, y + cell_h], fill=(0, 0, 0))
        draw.text((x + 6, y + thumb_h + 4), name, fill=(255, 255, 255), font=font)

    sheet_path = OUT_DIR / "_contact_sheet.png"
    sheet.save(sheet_path, format="PNG")
    print(f"Contact sheet: {sheet_path}")

    # Plus a tighter 5x4 sheet at smaller size for chat preview
    preview_w, preview_h = 192, 108
    pcell_w, pcell_h = preview_w, preview_h + 18
    preview = Image.new("RGB", (cols * pcell_w, rows * pcell_h), color=(20, 20, 20))
    pdraw = ImageDraw.Draw(preview)
    try:
        psmall = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        psmall = ImageFont.load_default()
    for i, (name, img) in enumerate(images):
        col = i % cols
        row = i // cols
        x = col * pcell_w
        y = row * pcell_h
        thumb = img.resize((preview_w, preview_h), Image.LANCZOS)
        preview.paste(thumb, (x, y))
        pdraw.text((x + 4, y + preview_h + 2), name, fill=(255, 255, 255), font=psmall)

    preview_path = OUT_DIR / "_contact_sheet_small.png"
    preview.save(preview_path, format="PNG", optimize=True)
    print(f"Small contact sheet: {preview_path}")


if __name__ == "__main__":
    main()
