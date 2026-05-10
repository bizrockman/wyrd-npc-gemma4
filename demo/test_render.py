"""
Image rendering test for the terminal demo.

Tries multiple backends and sizes to figure out what looks best in your
terminal. Run it, screenshot the output, and we'll pick the winner.

Usage:
    cd demo
    python test_render.py
    python test_render.py --image assets/kenji_pot.png --height 25

Or from the project root:
    python -X utf8 demo/test_render.py --image demo/assets/kenji_pot.png
"""
import argparse
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

try:
    from rich_pixels import Pixels
    HAS_RICH_PIXELS = True
except ImportError:
    HAS_RICH_PIXELS = False

try:
    from term_image.image import AutoImage, BlockImage
    HAS_TERM_IMAGE = True
except ImportError:
    HAS_TERM_IMAGE = False


def render_rich_pixels(console: Console, img_path: Path, target_height: int):
    """Render via rich-pixels at given character height (uses half-blocks)."""
    img = Image.open(img_path)
    # half-blocks: 2 image rows per terminal row
    target_pixel_height = target_height * 2
    aspect = img.width / img.height
    target_pixel_width = int(target_pixel_height * aspect)
    img_resized = img.resize((target_pixel_width, target_pixel_height), Image.LANCZOS)
    return Pixels.from_image(img_resized)


def render_ascii(img_path: Path, target_width: int = 60) -> str:
    """Pure ASCII fallback — no color, just brightness ramp."""
    chars = " .:-=+*#%@"
    img = Image.open(img_path).convert("L")
    aspect = img.width / img.height
    # ASCII chars are roughly 2:1 (twice as tall as wide)
    target_height = int(target_width / aspect / 2)
    img = img.resize((target_width, target_height), Image.LANCZOS)
    pixels = list(img.getdata())
    lines = []
    for y in range(target_height):
        line = ""
        for x in range(target_width):
            v = pixels[y * target_width + x]
            line += chars[v * (len(chars) - 1) // 255]
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None,
                        help="Path to image (relative to cwd, demo/, or project root)")
    parser.add_argument("--height", type=int, default=20,
                        help="Target height in terminal rows")
    args = parser.parse_args()

    console = Console()

    # Resolve image path against multiple anchors so it works regardless
    # of where the user invokes from.
    candidates = []
    if args.image:
        p = Path(args.image)
        candidates = [p, DEMO_DIR / p, PROJECT_DIR / p]
    else:
        # Default search order: demo/assets, then comic strip
        candidates = [
            DEMO_DIR / "assets" / "kenji_pot.png",
            DEMO_DIR / "assets" / "kenji_logo.png",
            DEMO_DIR / "assets" / "kenji.png",
            PROJECT_DIR / "comic" / "kenji_strip.png",
        ]

    img_path = next((c for c in candidates if c.exists()), None)
    if img_path is None:
        print("No image found. Tried:")
        for c in candidates:
            print(f"  {c}")
        print("\nDrop a PNG into demo/assets/ or pass --image PATH")
        sys.exit(1)

    img = Image.open(img_path)
    console.print(f"\n[bold]Source:[/bold] {img_path} ({img.width}x{img.height})\n")

    # ------------------------------------------------------------------
    # Test 1: rich-pixels at requested height
    # ------------------------------------------------------------------
    console.rule("[bold yellow]Test 1: rich-pixels (Unicode half-blocks)")
    if HAS_RICH_PIXELS:
        for h in [10, 20, 30]:
            console.print(f"\n[dim]Height: {h} rows[/dim]")
            try:
                pixels = render_rich_pixels(console, img_path, h)
                console.print(pixels)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    else:
        console.print("[red]rich-pixels not installed[/red]")

    # ------------------------------------------------------------------
    # Test 2: term-image (auto-detects best protocol)
    # ------------------------------------------------------------------
    console.rule("[bold yellow]Test 2: term-image (auto: sixel/kitty/iterm/fallback)")
    if HAS_TERM_IMAGE:
        try:
            ti = AutoImage.from_file(str(img_path))
            ti.set_size(height=args.height)
            console.print(f"[dim]Backend: {type(ti).__name__}, size: {ti.rendered_size}[/dim]")
            print(ti)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            try:
                ti = BlockImage.from_file(str(img_path))
                ti.set_size(height=args.height)
                console.print("[dim]Falling back to BlockImage[/dim]")
                print(ti)
            except Exception as e2:
                console.print(f"[red]BlockImage error: {e2}[/red]")
    else:
        console.print("[red]term-image not installed[/red]")

    # ------------------------------------------------------------------
    # Test 3: Pure ASCII fallback (no color, works everywhere)
    # ------------------------------------------------------------------
    console.rule("[bold yellow]Test 3: ASCII fallback (no color, max compat)")
    ascii_art = render_ascii(img_path, target_width=60)
    console.print(Panel(ascii_art, title="ASCII", box=box.SIMPLE))

    # ------------------------------------------------------------------
    # Test 4: Welcome panel mockup with image + text in Columns
    # ------------------------------------------------------------------
    console.rule("[bold yellow]Test 4: Welcome panel mockup (rich-pixels + Columns)")
    if HAS_RICH_PIXELS:
        try:
            portrait = render_rich_pixels(console, img_path, target_height=15)
            text = Text()
            text.append("Kenji's Ramen\n", style="bold yellow")
            text.append("Mendokoro Sato — Shinjuku Yokocho\n\n", style="dim")
            text.append("A narrow alley. Steam drifts from under the noren curtain.\n")
            text.append("You push through and sit down at the worn wooden counter.\n")
            text.append("The cook glances at you, then back at his pot.\n\n")
            text.append("Type freely or pick a suggestion. 'quit' to leave.\n", style="dim")

            cols = Columns([portrait, text], padding=(0, 2), expand=False)
            panel = Panel(
                cols,
                title="[yellow]麺処 佐藤[/yellow]",
                border_style="yellow",
                box=box.DOUBLE,
                padding=(1, 2),
            )
            console.print(panel)
        except Exception as e:
            console.print(f"[red]Mockup error: {e}[/red]")

    console.print("\n[dim]Done. Best looking option wins.[/dim]\n")


if __name__ == "__main__":
    main()
