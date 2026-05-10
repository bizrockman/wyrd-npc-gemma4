"""
Generate pixel-art assets for the terminal demo via Nano Banana Pro.

Pixel art renders far better in terminal half-block format than detailed
manga because the chunky pixels translate naturally to Unicode blocks.
Limited palettes and strong outlines stay readable at low resolution.

Usage:
    python generate_assets.py                  # generate all
    python generate_assets.py --asset alley    # one only
    python generate_assets.py --list           # show what would be generated

Requires FAL_KEY in .env (same setup as comic/generate_comic.py).
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import fal_client
import requests
from dotenv import load_dotenv
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parent
ASSETS_DIR = DEMO_DIR / "assets"
PROJECT_DIR = DEMO_DIR.parent

# .env can live in project root or in comic/ (shared with comic generator)
for env_path in [PROJECT_DIR / ".env", PROJECT_DIR / "comic" / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break

# fal_client expects FAL_KEY but the project uses FAL_API_KEY
if not os.getenv("FAL_KEY") and os.getenv("FAL_API_KEY"):
    os.environ["FAL_KEY"] = os.environ["FAL_API_KEY"]


# Target sizes for terminal rendering. Aggressive downsample pushes the
# image toward true pixel-art aesthetic. Half-block render uses 2 pixels
# per terminal row, so 72px height ≈ 36 terminal rows.
TARGET_SIZES = {
    "16:9": (128, 72),    # ~5 KB
    "1:1":  (96, 96),     # ~5 KB
    "3:4":  (72, 96),
    "4:3":  (96, 72),
}

# Color palette size — fewer colors = more "pixel art" look
PALETTE_COLORS = 16


def pixelize(image_bytes: bytes, aspect: str) -> bytes:
    """Aggressive pixel-art post-processing pipeline.

    1. Hard downsample to small target (NEAREST for chunky pixels, then
       LANCZOS would smooth them — we use a mix for balance)
    2. Posterize to 16 colors via median-cut quantization
    3. Save as compact PNG
    """
    target = TARGET_SIZES.get(aspect, (96, 96))
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Step 1: aggressive downsample. LANCZOS gives clean color reduction
    # without the "noise" you'd get from nearest-neighbor at this step.
    img = img.resize(target, Image.LANCZOS)

    # Step 2: hard color quantization for the pixel-art palette feel
    img = img.quantize(colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT,
                       dither=Image.Dither.NONE).convert("RGB")

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ----------------------------------------------------------------------------
# Style anchors — applied to every prompt for consistency
# ----------------------------------------------------------------------------

PIXEL_STYLE_BASE = (
    "16-bit JRPG pixel art style, limited palette of warm earth tones "
    "(browns, cream, ochre, deep red accents), strong dark outlines, "
    "chunky readable pixels, low resolution aesthetic, clear silhouettes, "
    "no anti-aliasing, no gradients, flat shading. "
    "Character design: middle-aged Japanese man, short dark hair with grey, "
    "white traditional cook jacket, white headband, calm reserved expression."
)

NEGATIVE = (
    "photorealistic, smooth shading, gradients, anti-aliasing, blurry, "
    "modern 3D, anime cel shading, complex background, busy details, "
    "text, watermark, signature"
)

# ----------------------------------------------------------------------------
# Asset definitions
# ----------------------------------------------------------------------------

ASSETS = {
    "intro_alley": {
        "filename": "intro_alley.png",
        "aspect": "16:9",
        "prompt": (
            "Pixel art establishing shot of a narrow Japanese yokocho alley "
            "at night. Two glowing red paper lanterns hanging from above, "
            "a small ramen shop with white noren curtains, wooden counter "
            "and stools visible through the entrance, steam rising. "
            "Brick or concrete pavement, wooden building facades on both "
            "sides, warm lantern glow as the only light source. "
            "Cinematic wide composition, no characters in frame. "
            f"{PIXEL_STYLE_BASE}"
        ),
    },
    "kenji_idle": {
        "filename": "kenji_idle.png",
        "aspect": "1:1",
        "prompt": (
            "Pixel art portrait of a middle-aged Japanese ramen cook standing "
            "behind his wooden counter, arms relaxed at his sides, calm "
            "neutral expression, looking forward. The counter and a hint "
            "of the kitchen behind him are visible. Centered composition, "
            "character takes about 60% of the frame height. "
            f"{PIXEL_STYLE_BASE}"
        ),
    },
    "kenji_cooking": {
        "filename": "kenji_cooking.png",
        "aspect": "1:1",
        "prompt": (
            "Pixel art portrait of a middle-aged Japanese ramen cook stirring "
            "a large steaming pot of broth with a long ladle, focused "
            "expression, slightly hunched over the pot. Pixel-art steam "
            "rising in clearly defined puffs. The wooden counter is in front "
            "of him. Centered composition. "
            f"{PIXEL_STYLE_BASE}"
        ),
    },
    "kenji_serving": {
        "filename": "kenji_serving.png",
        "aspect": "1:1",
        "prompt": (
            "Pixel art portrait of a middle-aged Japanese ramen cook placing "
            "a steaming bowl of ramen on the wooden counter with both hands, "
            "calm focused expression, leaning slightly forward. The bowl is "
            "in clear pixel detail with chashu and an egg visible. Centered "
            "composition. "
            f"{PIXEL_STYLE_BASE}"
        ),
    },
    "kenji_wiping": {
        "filename": "kenji_wiping.png",
        "aspect": "1:1",
        "prompt": (
            "Pixel art portrait of a middle-aged Japanese ramen cook wiping "
            "the wooden counter with a cloth in slow methodical motion, "
            "looking down at the counter, neutral expression. Centered "
            "composition. "
            f"{PIXEL_STYLE_BASE}"
        ),
    },
    "kenji_portrait": {
        "filename": "kenji_portrait.png",
        "aspect": "1:1",
        "prompt": (
            "Pixel art bust portrait of a middle-aged Japanese ramen cook, "
            "head and shoulders, white headband, white cook jacket, calm "
            "weathered face, looking slightly off-camera. Plain warm cream "
            "background. Tight composition, character fills 80% of frame. "
            f"{PIXEL_STYLE_BASE}"
        ),
    },
}


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def generate(asset_key: str, asset_def: dict, force: bool = False) -> bool:
    """Generate one asset via Nano Banana Pro. Returns True on success."""
    out_path = ASSETS_DIR / asset_def["filename"]
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to regenerate)")
        return True

    print(f"  [gen ] {out_path.name} ({asset_def['aspect']})...")
    start = time.time()
    try:
        result = fal_client.subscribe(
            "fal-ai/nano-banana-pro",
            arguments={
                "prompt": asset_def["prompt"],
                "num_images": 1,
                "aspect_ratio": asset_def["aspect"],
                "resolution": "1K",
                "output_format": "png",
            },
            with_logs=False,
        )
        url = result["images"][0]["url"]
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        original_kb = len(resp.content) // 1024

        # Pixelize: aggressive downsample + 16-color quantize for pixel-art look
        small = pixelize(resp.content, asset_def["aspect"])
        out_path.write_bytes(small)
        elapsed = time.time() - start
        print(f"  [ok  ] {out_path.name} saved ({len(small)//1024} KB "
              f"down from {original_kb} KB, {elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  [FAIL] {out_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", help="Generate only this asset (key from ASSETS dict)")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if file exists")
    parser.add_argument("--list", action="store_true",
                        help="List assets without generating")
    parser.add_argument("--repixelize", action="store_true",
                        help="Re-run the pixel-art post-process on existing files (no API call)")
    args = parser.parse_args()

    if args.list:
        print("Available assets:")
        for key, ad in ASSETS.items():
            exists = "✓" if (ASSETS_DIR / ad["filename"]).exists() else " "
            print(f"  [{exists}] {key:20s} → {ad['filename']:25s} ({ad['aspect']})")
        return

    if args.repixelize:
        targets = {args.asset: ASSETS[args.asset]} if args.asset else ASSETS
        for key, ad in targets.items():
            p = ASSETS_DIR / ad["filename"]
            if not p.exists():
                print(f"  [skip] {p.name} not found")
                continue
            original = p.read_bytes()
            new = pixelize(original, ad["aspect"])
            p.write_bytes(new)
            print(f"  [pix ] {p.name}: {len(original)//1024} KB → {len(new)//1024} KB")
        return

    if not os.getenv("FAL_KEY"):
        print("Error: FAL_KEY not set in .env or environment")
        sys.exit(1)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.asset: ASSETS[args.asset]} if args.asset else ASSETS
    if args.asset and args.asset not in ASSETS:
        print(f"Unknown asset: {args.asset}. Available: {', '.join(ASSETS)}")
        sys.exit(1)

    print(f"Generating {len(targets)} asset(s) → {ASSETS_DIR}")
    print("=" * 60)
    failures = 0
    for key, ad in targets.items():
        if not generate(key, ad, force=args.force):
            failures += 1

    print("=" * 60)
    print(f"Done. {len(targets) - failures}/{len(targets)} successful.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
