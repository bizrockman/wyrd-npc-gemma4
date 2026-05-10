"""
Scene sweep: 20 distinct scenes from Kenji's world, each generated as
pixel art. This time we vary the SUBJECT, not just the style.

Run after .venv-pixelart is set up:
    .venv-pixelart\\Scripts\\activate
    python demo/sweep_scenes.py
"""
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

DEMO_DIR = Path(__file__).resolve().parent
OUT_DIR = DEMO_DIR / "assets" / "scenes"

# Style anchor used in every prompt — keeps the pixel-art aesthetic consistent
STYLE = "pixel art, 16-bit JRPG style, retro game sprite, vibrant warm palette, dark outlines, no anti-aliasing"

NEGATIVE = (
    "photorealistic, smooth shading, blurry, anti-aliasing, gradients, "
    "modern 3D render, cel shading, text, watermark, signature"
)

# 20 distinct scenes from the Kenji universe
SCENES = [
    # --- Establishing / world (4) ---
    ("01_alley_night",     "narrow Tokyo yokocho alley at night, multiple shop signs and red paper lanterns, "
                           "no people, atmospheric mood",                       "16:9"),
    ("02_alley_rain",      "narrow yokocho alley in light rain at night, reflections on wet stones, "
                           "glowing lanterns, no people",                       "16:9"),
    ("03_shop_exterior",   "small ramen shop facade with white noren curtain saying 麺処, "
                           "two red lanterns, glowing window, view from across the alley", "16:9"),
    ("04_shop_morning",    "ramen shop exterior in soft morning light, noren curtain closed, "
                           "no lanterns lit, no people, quiet street",          "16:9"),

    # --- Interior wide (3) ---
    ("05_interior_empty",  "small ramen shop interior, eight wooden stools at a wooden counter, "
                           "no people, kitchen visible behind counter, warm light", "16:9"),
    ("06_interior_kenji",  "small ramen shop interior, ramen cook in white headband and jacket "
                           "alone behind the counter, wiping it with a cloth",  "16:9"),
    ("07_interior_busy",   "small ramen shop interior at dinner, three customers at the counter "
                           "eating, ramen cook in white jacket working at the stove", "16:9"),

    # --- Kenji portraits / actions (5) ---
    ("08_kenji_portrait",  "middle-aged Japanese ramen cook in white headband, calm weathered face, "
                           "head and shoulders, plain warm background",         "1:1"),
    ("09_kenji_stirring",  "middle-aged Japanese ramen cook stirring large pot of broth with long ladle, "
                           "steam rising, focused expression",                  "1:1"),
    ("10_kenji_serving",   "middle-aged Japanese ramen cook placing steaming bowl of ramen on wooden counter "
                           "with both hands",                                   "1:1"),
    ("11_kenji_wiping",    "middle-aged Japanese ramen cook wiping wooden counter with a cloth, "
                           "looking down at his work",                          "1:1"),
    ("12_kenji_thinking",  "middle-aged Japanese ramen cook standing still behind counter, "
                           "arms crossed, looking off to the side",             "1:1"),

    # --- Object close-ups (4) ---
    ("13_bowl_topdown",    "top-down view of a tonkotsu ramen bowl, chashu pork slices, soft-boiled egg, "
                           "noodles, green onions, dark broth, on wooden counter", "1:1"),
    ("14_pot_steam",       "large steaming metal pot of pork bone broth on a stove, "
                           "steam rising in clear puffs, close-up",             "1:1"),
    ("15_kenbaiki",        "Japanese ticket vending machine (kenbaiki) with menu buttons, "
                           "by the entrance of a small ramen shop, close-up",   "3:4"),
    ("16_noren_close",     "white noren curtain with kanji 麺処, hanging in shop entrance, "
                           "close-up, soft warm light from inside",             "3:4"),

    # --- Atmosphere (4) ---
    ("17_lantern_glow",    "single red Japanese paper lantern hanging in dark alley at night, "
                           "warm glow, kanji ラーメン on it",                    "3:4"),
    ("18_counter_seat",    "first-person view from a stool at the wooden counter, looking at "
                           "the kitchen and ramen cook working, steam rising",  "16:9"),
    ("19_late_night",      "ramen shop interior at closing time, lights dim, cook wiping down "
                           "the counter alone, empty stools",                   "16:9"),
    ("20_alley_dawn",      "yokocho alley at dawn, soft pink and blue sky between rooftops, "
                           "shops still closed, atmospheric quiet",             "16:9"),
]

ASPECT_DIMS = {
    "16:9": (1024, 576),
    "1:1":  (768, 768),
    "3:4":  (576, 768),
    "4:3":  (768, 576),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print("Loading SDXL + pixel-art-xl LoRA...")

    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.load_lora_weights("nerijs/pixel-art-xl")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"\nGenerating {len(SCENES)} scenes...")
    print("=" * 60)

    images = []
    start_total = time.time()
    for i, (name, scene, aspect) in enumerate(SCENES, 1):
        out_path = OUT_DIR / f"{name}.png"
        w, h = ASPECT_DIMS[aspect]
        full_prompt = f"{STYLE}, {scene}"
        print(f"  [{i:2d}/{len(SCENES)}] {name} ({aspect})", flush=True)

        # Different seed per scene so they really differ
        gen = torch.Generator(device="cuda").manual_seed(1000 + i)
        img = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=8,
            guidance_scale=1.5,
            width=w,
            height=h,
            generator=gen,
        ).images[0]
        img.save(out_path, format="PNG", optimize=True)
        images.append((name, aspect, img))

    elapsed = time.time() - start_total
    print(f"\nGeneration done in {elapsed:.1f}s ({elapsed/len(SCENES):.1f}s avg)")

    # Contact sheet — group by aspect for clean grid
    print("Building contact sheet...")
    cols, rows = 5, 4
    thumb_w, thumb_h = 256, 192  # cell size; we'll fit each thumb to its aspect
    label_h = 22
    cell_w, cell_h = thumb_w, thumb_h + label_h
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (name, aspect, img) in enumerate(images):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = row * cell_h
        # Fit each image into thumb_w x thumb_h preserving aspect, letterbox
        ar = img.width / img.height
        if ar >= thumb_w / thumb_h:
            new_w = thumb_w
            new_h = int(thumb_w / ar)
        else:
            new_h = thumb_h
            new_w = int(thumb_h * ar)
        thumb = img.resize((new_w, new_h), Image.LANCZOS)
        # Center inside the cell
        ox = x + (thumb_w - new_w) // 2
        oy = y + (thumb_h - new_h) // 2
        sheet.paste(thumb, (ox, oy))
        draw.rectangle([x, y + thumb_h, x + cell_w, y + cell_h], fill=(0, 0, 0))
        draw.text((x + 4, y + thumb_h + 3), name, fill=(255, 255, 255), font=font)

    sheet_path = OUT_DIR / "_scenes_contact_sheet.png"
    sheet.save(sheet_path, format="PNG")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
