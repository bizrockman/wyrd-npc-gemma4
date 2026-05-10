"""
Local pixel-art asset generation via SDXL + pixel-art-xl LoRA.

Uses HuggingFace diffusers + nerijs/pixel-art-xl LoRA for native pixel-art
output (no post-process tricks needed). Requires the .venv-pixelart venv
with torch+CUDA and diffusers installed.

Usage (after activating .venv-pixelart):
    python demo/generate_assets_local.py --list
    python demo/generate_assets_local.py --asset intro_alley
    python demo/generate_assets_local.py
    python demo/generate_assets_local.py --steps 12 --guidance 2.0

The first run will download SDXL base (~7GB) + LoRA (~50MB) to the
HuggingFace cache (~/.cache/huggingface).
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parent
ASSETS_DIR = DEMO_DIR / "assets"
PROJECT_DIR = DEMO_DIR.parent

# ----------------------------------------------------------------------------
# Asset definitions (style anchors woven into each prompt)
# ----------------------------------------------------------------------------

# nerijs/pixel-art-xl trigger word: "pixel art"
# We wrap our character description AROUND it with strong pixel-art language
PIXEL_PROMPT_TEMPLATE = (
    "pixel art, {scene}, "
    "16-bit JRPG style, retro game sprite, NES SNES aesthetic, "
    "limited palette of warm earth tones, dark outlines, "
    "no anti-aliasing, no smooth gradients, chunky readable pixels"
)

NEGATIVE_PROMPT = (
    "photorealistic, smooth shading, blurry, anti-aliasing, gradients, "
    "modern 3D render, anime cel shading, complex texture detail, "
    "text, watermark, signature, logo"
)

ASSETS = {
    "intro_alley": {
        "filename": "intro_alley.png",
        "size": (1024, 576),  # 16:9 SDXL-friendly
        "scene": (
            "narrow Japanese yokocho alley at night, two glowing red paper "
            "lanterns hanging from above, small ramen shop with white noren "
            "curtains, wooden counter and stools, steam rising, brick path, "
            "warm lantern lighting, no characters"
        ),
    },
    "kenji_idle": {
        "filename": "kenji_idle.png",
        "size": (768, 768),  # 1:1
        "scene": (
            "middle-aged Japanese ramen cook in white headband and white "
            "cook jacket, standing behind a wooden counter, arms relaxed, "
            "calm neutral expression, looking forward, kitchen behind him"
        ),
    },
    "kenji_cooking": {
        "filename": "kenji_cooking.png",
        "size": (768, 768),
        "scene": (
            "middle-aged Japanese ramen cook in white headband and white "
            "cook jacket, stirring large pot of broth with long ladle, "
            "focused expression, steam rising in clear puffs, wooden counter"
        ),
    },
    "kenji_serving": {
        "filename": "kenji_serving.png",
        "size": (768, 768),
        "scene": (
            "middle-aged Japanese ramen cook in white headband and white "
            "cook jacket, placing steaming bowl of ramen on wooden counter "
            "with both hands, calm focused expression, bowl with chashu and egg"
        ),
    },
    "kenji_wiping": {
        "filename": "kenji_wiping.png",
        "size": (768, 768),
        "scene": (
            "middle-aged Japanese ramen cook in white headband and white "
            "cook jacket, wiping wooden counter with cloth, looking down, "
            "neutral focused expression"
        ),
    },
    "kenji_portrait": {
        "filename": "kenji_portrait.png",
        "size": (768, 768),
        "scene": (
            "middle-aged Japanese ramen cook bust portrait, white headband, "
            "white cook jacket, calm weathered face, looking slightly off-camera, "
            "plain warm cream background"
        ),
    },
}


# ----------------------------------------------------------------------------
# Pipeline setup
# ----------------------------------------------------------------------------

def load_pipeline(verbose: bool = True):
    """Load SDXL base + pixel-art-xl LoRA. Returns the diffusers pipeline."""
    from diffusers import DiffusionPipeline

    if verbose:
        print("Loading SDXL base (this can take a while on first run)...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    if verbose:
        print("Loading pixel-art-xl LoRA from nerijs/pixel-art-xl...")
    pipe.load_lora_weights("nerijs/pixel-art-xl")

    # SDXL benefits from a faster scheduler when used with this LoRA
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe


# ----------------------------------------------------------------------------
# Pixelize post-process (mild, the LoRA does most of the work)
# ----------------------------------------------------------------------------

TARGET_SIZES = {
    "16:9": (128, 72),
    "1:1":  (96, 96),
}

PALETTE_COLORS = 16


def pixelize(img: Image.Image, aspect: str) -> bytes:
    """Final terminal-friendly pixelize: downsample + 16-color palette."""
    target = TARGET_SIZES.get(aspect, (96, 96))
    img = img.convert("RGB").resize(target, Image.LANCZOS)
    img = img.quantize(colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT,
                       dither=Image.Dither.NONE).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def aspect_for(size: tuple[int, int]) -> str:
    w, h = size
    if abs(w / h - 16 / 9) < 0.05:
        return "16:9"
    if abs(w / h - 1.0) < 0.05:
        return "1:1"
    return "1:1"


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def generate(pipe, asset_key: str, asset_def: dict, steps: int, guidance: float,
             seed: int | None, force: bool = False, also_small: bool = False) -> bool:
    out_path = ASSETS_DIR / asset_def["filename"]
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (--force to regenerate)")
        return True

    prompt = PIXEL_PROMPT_TEMPLATE.format(scene=asset_def["scene"])
    w, h = asset_def["size"]

    print(f"  [gen ] {out_path.name} ({w}x{h}, {steps} steps)...")
    start = time.time()
    try:
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=w,
            height=h,
            generator=generator,
        )
        full_img = result.images[0]

        # The LoRA already generates pixel-art aesthetic; saving the full-res
        # output as the main asset preserves the LoRA's intended palette and
        # pixel grid. The terminal renderer in test_render.py / kenji_terminal.py
        # downsamples for display at render time.
        full_img.save(out_path, format="PNG", optimize=True)

        small_note = ""
        if also_small:
            small_path = ASSETS_DIR / "_small" / asset_def["filename"]
            small_path.parent.mkdir(parents=True, exist_ok=True)
            small_bytes = pixelize(full_img, aspect_for(asset_def["size"]))
            small_path.write_bytes(small_bytes)
            small_note = f", small at {small_path.relative_to(PROJECT_DIR)}"

        elapsed = time.time() - start
        size_kb = out_path.stat().st_size // 1024
        print(f"  [ok  ] {out_path.name} saved ({size_kb} KB{small_note}, {elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  [FAIL] {out_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", help="Generate only this asset key")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--steps", type=int, default=8,
                        help="Inference steps (default 8 for fast)")
    parser.add_argument("--guidance", type=float, default=1.5,
                        help="Guidance scale (default 1.5; pixel-art LoRA likes low)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--also-small", action="store_true",
                        help="Also save aggressively pixelized version under assets/_small/")
    args = parser.parse_args()

    if args.list:
        print("Available assets:")
        for key, ad in ASSETS.items():
            exists = "✓" if (ASSETS_DIR / ad["filename"]).exists() else " "
            w, h = ad["size"]
            print(f"  [{exists}] {key:20s} -> {ad['filename']:25s} ({w}x{h})")
        return

    if not torch.cuda.is_available():
        print("Error: CUDA not available. Activate .venv-pixelart and verify torch+CUDA.")
        sys.exit(1)

    print(f"CUDA: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.asset: ASSETS[args.asset]} if args.asset else ASSETS
    if args.asset and args.asset not in ASSETS:
        print(f"Unknown asset: {args.asset}. Available: {', '.join(ASSETS)}")
        sys.exit(1)

    pipe = load_pipeline(verbose=True)
    print(f"\nGenerating {len(targets)} asset(s) -> {ASSETS_DIR}")
    print(f"Steps: {args.steps}, Guidance: {args.guidance}, "
          f"Seed: {args.seed or 'random'}")
    print("=" * 60)

    failures = 0
    for key, ad in targets.items():
        if not generate(pipe, key, ad, args.steps, args.guidance,
                        args.seed, force=args.force, also_small=args.also_small):
            failures += 1

    print("=" * 60)
    print(f"Done. {len(targets) - failures}/{len(targets)} successful.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
