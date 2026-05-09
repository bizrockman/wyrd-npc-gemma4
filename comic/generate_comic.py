"""
Generate comic strip images from dialogue YAML files.

Reads dialogue strips produced by generate_dialogue.py, converts each
turn into panel prompts with the selected art style, and generates
images via fal.ai.

Usage:
    python generate_comic.py strips/same_question_20260509_194910.yaml
    python generate_comic.py strips/same_question_20260509_194910.yaml --style franco_belgian
    python generate_comic.py strips/same_question_20260509_194910.yaml --style manga --model nano-banana-pro
    python generate_comic.py --list-styles

Models available:
    nano-banana-pro   Gemini-based, best quality ($0.15/image)
    flux-schnell      Fastest, good for drafts ($0.003/image)
    flux-2-pro        High quality diffusion ($0.03/megapixel)
    imagen3           Google Imagen 3 ($0.05/image)
    imagen3-fast      Imagen 3 low latency ($0.03/image)
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import fal_client
import yaml
from dotenv import load_dotenv

from styles import STYLES, CHARACTER, SETTING, CAMERA, build_panel_prompt, build_full_page_prompt, list_styles


# ---------------------------------------------------------------------------
# fal.ai model endpoints
# ---------------------------------------------------------------------------

FAL_MODELS = {
    "nano-banana-pro": {
        "endpoint": "fal-ai/nano-banana-pro",
        "params": lambda prompt, neg, aspect: {
            "prompt": prompt,
            "num_images": 1,
            "aspect_ratio": aspect,
            "resolution": "1K",
            "output_format": "png",
        },
    },
    "flux-schnell": {
        "endpoint": "fal-ai/flux/schnell",
        "params": lambda prompt, neg, aspect: {
            "prompt": prompt,
            "image_size": aspect_to_flux_size(aspect),
            "num_images": 1,
        },
    },
    "flux-2-pro": {
        "endpoint": "fal-ai/flux-2-pro",
        "params": lambda prompt, neg, aspect: {
            "prompt": prompt,
            "image_size": aspect_to_flux_size(aspect),
            "num_images": 1,
        },
    },
    "imagen3": {
        "endpoint": "fal-ai/imagen3",
        "params": lambda prompt, neg, aspect: {
            "prompt": prompt,
            "aspect_ratio": aspect,
            "num_images": 1,
        },
    },
    "imagen3-fast": {
        "endpoint": "fal-ai/imagen3/fast",
        "params": lambda prompt, neg, aspect: {
            "prompt": prompt,
            "aspect_ratio": aspect,
            "num_images": 1,
        },
    },
}


def aspect_to_flux_size(aspect: str) -> str:
    """Convert aspect ratio string to Flux image_size enum."""
    mapping = {
        "4:3": "landscape_4_3",
        "3:4": "portrait_4_3",
        "16:9": "landscape_16_9",
        "9:16": "portrait_16_9",
        "1:1": "square",
        "3:2": "landscape_4_3",  # closest
    }
    return mapping.get(aspect, "landscape_4_3")


# ---------------------------------------------------------------------------
# Panel planning
# ---------------------------------------------------------------------------

def plan_panels(strip: dict) -> list[dict]:
    """
    Convert a dialogue strip into a sequence of panel specifications.

    Each panel gets: scene_description, characters, camera angle, and
    whether it's a dialogue panel (needs speech bubble) or establishing.
    """
    panels = []

    # Panel 1: Establishing shot (always)
    panels.append({
        "id": "establishing",
        "scene_description": (
            "Exterior view of a tiny ramen shop in a narrow yokocho alley. "
            "Steam drifts from under the noren curtain. Paper lanterns glow "
            "warmly. A few people pass in the background."
        ),
        "characters": [],
        "camera": "establishing",
        "dialogue": None,
        "speaker": None,
        "aspect": "16:9",
    })

    # Generate panels from dialogue turns
    for scene in strip.get("scenes", []):
        scene_id = scene["id"]
        characters_in_scene = ["kenji"]
        if scene_id in CHARACTER:
            characters_in_scene.append(scene_id)

        for i, turn in enumerate(scene.get("turns", [])):
            response = turn.get("response", "")

            # Extract scene description from **scene** markers
            scene_desc = extract_scene_description(response)
            dialogue_text = extract_dialogue(response)

            # Determine camera based on position in scene
            if i == 0:
                camera = "enter" if i == 0 and len(panels) <= 2 else "dialogue"
            elif dialogue_text and len(dialogue_text) < 30:
                camera = "reaction"  # Short responses = close-up
            else:
                camera = "dialogue"

            # Customer speaks panel (if the input is interesting)
            user_input = turn.get("input", "")
            if user_input and not user_input.startswith("*"):
                panels.append({
                    "id": f"{scene_id}_customer_{i}",
                    "scene_description": (
                        f"Customer speaking to the ramen shop owner across the counter. "
                        f"{scene_desc or 'Warm ambient lighting, steam in the air.'}"
                    ),
                    "characters": list(reversed(characters_in_scene)),  # customer focus
                    "camera": "dialogue",
                    "dialogue": clean_for_bubble(user_input),
                    "speaker": "customer",
                    "aspect": "4:3",
                })

            # Kenji responds panel
            panels.append({
                "id": f"{scene_id}_kenji_{i}",
                "scene_description": scene_desc or "Kenji behind the counter, steam rising from the pot",
                "characters": characters_in_scene,
                "camera": camera,
                "dialogue": clean_for_bubble(dialogue_text) if dialogue_text else None,
                "speaker": "kenji",
                "aspect": "4:3",
            })

    return panels


def extract_scene_description(response: str) -> str:
    """Extract scene markers from model output."""
    # Match **scene** followed by text until the next line or quote
    matches = re.findall(r'\*\*scene\*\*\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
    if matches:
        return " ".join(m.strip() for m in matches)
    return ""


def extract_dialogue(response: str) -> str:
    """Extract spoken dialogue from model output (everything that's not a scene marker)."""
    # Remove scene markers
    text = re.sub(r'\*\*scene\*\*[^"]*?(?="|$)', '', response, flags=re.IGNORECASE)
    # Extract quoted speech
    quotes = re.findall(r'"([^"]+)"', text)
    if quotes:
        return " ".join(quotes)
    # Fallback: remove all scene markers and return remaining text
    text = re.sub(r'\*\*scene\*\*.*?(?:\n|$)', '', response, flags=re.IGNORECASE).strip()
    return text if text else ""


def clean_for_bubble(text: str) -> str:
    """Clean text for use in a speech bubble."""
    if not text:
        return ""
    text = text.strip().strip('"').strip("*").strip()
    text = re.sub(r'\*\*scene\*\*.*', '', text, flags=re.IGNORECASE).strip()
    return text


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_panel_image(
    panel: dict,
    style: str,
    fal_model: str,
    output_dir: Path,
    dry_run: bool = False,
) -> str | None:
    """Generate a single panel image and save to disk."""
    prompt_data = build_panel_prompt(
        scene_description=panel["scene_description"],
        characters=panel["characters"],
        camera=panel["camera"],
        style=style,
    )

    model_config = FAL_MODELS[fal_model]
    api_params = model_config["params"](
        prompt_data["prompt"],
        prompt_data["negative_prompt"],
        panel.get("aspect", "4:3"),
    )

    # Add negative prompt if model supports it
    if fal_model in ("flux-2-pro", "flux-schnell"):
        pass  # Flux doesn't use negative prompts
    elif "negative_prompt" in prompt_data:
        api_params["negative_prompt"] = prompt_data["negative_prompt"]

    panel_id = panel["id"]
    print(f"  Panel {panel_id}:")

    if dry_run:
        print(f"    [DRY RUN] Prompt: {prompt_data['prompt'][:120]}...")
        print(f"    [DRY RUN] Model: {model_config['endpoint']}")
        return None

    print(f"    Generating with {fal_model}...")
    start = time.time()

    try:
        result = fal_client.subscribe(
            model_config["endpoint"],
            arguments=api_params,
            with_logs=False,
        )
        elapsed = time.time() - start

        # Extract image URL
        images = result.get("images", [])
        if not images:
            print(f"    ERROR: No images returned")
            return None

        image_url = images[0].get("url", "")
        print(f"    Done ({elapsed:.1f}s): {image_url[:80]}...")

        # Download image
        import httpx
        img_path = output_dir / f"{panel_id}.png"
        resp = httpx.get(image_url)
        img_path.write_bytes(resp.content)
        print(f"    Saved: {img_path}")
        return str(img_path)

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate comic strip images from dialogue")
    parser.add_argument("strip_file", nargs="?", help="Dialogue YAML file from generate_dialogue.py")
    parser.add_argument("--style", default="manga", choices=STYLES.keys(), help="Art style preset")
    parser.add_argument("--model", default="nano-banana-pro", choices=FAL_MODELS.keys(), help="Image model")
    parser.add_argument("--list-styles", action="store_true", help="Show available styles")
    parser.add_argument("--list-models", action="store_true", help="Show available image models")
    parser.add_argument("--full-page", action="store_true", help="Generate entire strip as single image (recommended)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without generating")
    parser.add_argument("--max-panels", type=int, default=None, help="Limit number of panels (panel-by-panel mode)")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    if args.list_styles:
        print("Available styles:")
        for s in list_styles():
            print(f"  {s['key']:20s} {s['name']:20s} {s['description']}")
        return

    if args.list_models:
        print("Available models:")
        for k, v in FAL_MODELS.items():
            print(f"  {k:20s} {v['endpoint']}")
        return

    if not args.strip_file:
        parser.error("strip_file is required (or use --list-styles / --list-models)")

    # Load environment
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    # fal-client expects FAL_KEY
    fal_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY") or os.environ.get("FAL-API-KEY")
    if fal_key:
        os.environ["FAL_KEY"] = fal_key
    elif not args.dry_run:
        print("ERROR: No FAL API key found. Set FAL_KEY, FAL_API_KEY, or FAL-API-KEY in .env")
        sys.exit(1)

    # Load strip
    strip_path = Path(args.strip_file)
    if not strip_path.is_absolute():
        strip_path = Path(__file__).parent / strip_path
    strip = yaml.safe_load(strip_path.read_text(encoding="utf-8"))

    print(f"Strip: {strip.get('title', 'Unknown')}")
    print(f"Style: {args.style} ({STYLES[args.style]['name']})")
    print(f"Model: {args.model}")
    print(f"Mode:  {'full-page' if args.full_page else 'panel-by-panel'}")
    print()

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "full" if args.full_page else "panels"
        output_dir = Path(__file__).parent / "output" / f"{strip_path.stem}_{args.style}_{mode_tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.full_page:
        # --- Full-page mode: single image with all panels ---
        prompt_data = build_full_page_prompt(strip, style=args.style)

        # Save prompt for reproducibility
        prompt_path = output_dir / "prompt.txt"
        prompt_path.write_text(prompt_data["prompt"], encoding="utf-8")

        if args.dry_run:
            print("[DRY RUN] Full-page prompt:")
            print(prompt_data["prompt"])
            return

        model_config = FAL_MODELS[args.model]
        api_params = model_config["params"](
            prompt_data["prompt"],
            prompt_data.get("negative_prompt", ""),
            "16:9",
        )

        print("Generating full-page strip...")
        start = time.time()
        try:
            result = fal_client.subscribe(
                model_config["endpoint"],
                arguments=api_params,
                with_logs=False,
            )
            elapsed = time.time() - start
            images = result.get("images", [])
            if not images:
                print("ERROR: No images returned")
                sys.exit(1)

            image_url = images[0].get("url", "")
            print(f"Done ({elapsed:.1f}s): {image_url[:80]}...")

            import httpx
            img_path = output_dir / "strip.png"
            resp = httpx.get(image_url)
            img_path.write_bytes(resp.content)
            print(f"Saved: {img_path}")

        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        manifest = {
            "strip": strip.get("title"),
            "style": args.style,
            "model": args.model,
            "mode": "full-page",
            "generated_at": datetime.now().isoformat(),
            "image": str(img_path),
        }

    else:
        # --- Panel-by-panel mode ---
        panels = plan_panels(strip)
        if args.max_panels:
            panels = panels[:args.max_panels]
        print(f"Panels: {len(panels)}")
        print()

        plan_path = output_dir / "panel_plan.yaml"
        plan_path.write_text(
            yaml.dump(panels, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

        results = []
        for panel in panels:
            path = generate_panel_image(
                panel=panel,
                style=args.style,
                fal_model=args.model,
                output_dir=output_dir,
                dry_run=args.dry_run,
            )
            results.append({"panel_id": panel["id"], "image_path": path, "dialogue": panel.get("dialogue")})

        manifest = {
            "strip": strip.get("title"),
            "style": args.style,
            "model": args.model,
            "mode": "panel-by-panel",
            "generated_at": datetime.now().isoformat(),
            "panels": results,
        }

    manifest_path = output_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.dump(manifest, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
