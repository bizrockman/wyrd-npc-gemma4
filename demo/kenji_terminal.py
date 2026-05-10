"""
Interactive terminal demo for Kenji's Ramen.

Talk to Kenji at the counter. The demo suggests conversation options
based on what a customer might naturally say to a ramen shop owner -
generated independently from Kenji's character to avoid leaking
his internal state.

Usage:
    python kenji_terminal.py
    python kenji_terminal.py --no-suggestions

Requires: pip install requests rich pyyaml
"""

import argparse
import json
import re
import os
import sys
import time
from pathlib import Path

import requests

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' for a better terminal experience (pip install rich)")

try:
    from rich_pixels import Pixels
    from PIL import Image
    HAS_PIXELS = True
except ImportError:
    HAS_PIXELS = False

try:
    from term_image.image import BlockImage
    HAS_TERM_IMAGE = True
except ImportError:
    HAS_TERM_IMAGE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Models offered in the picker (in preference order)
# default_mode: "dialogue" (small models — capacity reserved for accuracy)
#               "scene"    (larger models — scenes add atmosphere)
PREFERRED_MODELS = [
    ("gemma4:e2b", "Gemma 4 e2b  (2.3B eff. — fast)            [dialogue-only]", "dialogue"),
    ("gemma4:e4b", "Gemma 4 e4b  (4.3B eff. — balanced)        [scene+dialogue]", "scene"),
    ("gemma4:12b", "Gemma 4 12b  (12B — strong)                [scene+dialogue]", "scene"),
    ("gemma4:26b", "Gemma 4 26b  (26B — best quality, PLE)     [scene+dialogue]", "scene"),
    ("gemma4:27b", "Gemma 4 27b  (27B — best quality, PLE)     [scene+dialogue]", "scene"),
    ("gemma4:31b", "Gemma 4 31b  (31B — dense, no PLE)         [scene+dialogue]", "scene"),
]

# Spec files
SPEC_FILES = {
    "scene":    "kenji_sato.en.yaml",            # scene+dialogue (default)
    "dialogue": "kenji_sato.dialogue_only.en.yaml",  # dialogue-only (small models)
}

SUGGESTION_SYSTEM = """\
You are a conversation coach helping a customer at a small ramen shop \
in Tokyo. The customer is sitting at the counter. The cook is a quiet, \
middle-aged Japanese man.

The visit follows a natural flow:
1. ARRIVING: sit down, look at menu, ask what's good, order at the ticket machine
2. WAITING: small talk, observe the cook working, ask about the place
3. EATING: appreciate the food, ask about ingredients or technique
4. FINISHING: thank him, ask about the neighborhood, say goodbye

Based on the conversation so far, figure out which PHASE the customer \
is in and suggest 4 things that fit that moment. Don't suggest ordering \
if they already ordered. Don't suggest food questions if they're already eating.

Rules:
- Keep each suggestion under 15 words
- Sound natural, not scripted
- Match the phase — after ordering, suggest waiting/chatting, not more ordering
- Don't reference anything the cook hasn't mentioned
- Don't ask about sensitive topics (money, past career, family problems)
- Write as the customer would speak, casual and friendly
- Output ONLY a JSON array of 4 strings, nothing else

Example — just sat down:
["What do you recommend?", "Busy night tonight?", \
"You been doing this long?", "This smells amazing, what's in it?"]

Example — food just arrived:
["This looks incredible.", "Is that black garlic oil on top?", \
"How long did you train to make this?", "Can I get a beer with this?"]"""


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def get_available_models() -> list[str]:
    """Get list of all models available in Ollama."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.ConnectionError:
        return []


def check_ollama(model: str) -> bool:
    """Check if Ollama is running and model is available."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if model not in models:
            print(f"Error: Model '{model}' not found in Ollama.")
            print(f"Available: {', '.join(models)}")
            print(f"\nPull it with: ollama pull {model}")
            return False
        return True
    except requests.ConnectionError:
        print("Error: Ollama is not running. Start it with: ollama serve")
        return False


def pick_model(console=None) -> tuple[str, str]:
    """Interactive model picker. Returns (model_name, default_mode)."""
    available = get_available_models()
    if not available:
        print("Error: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # Filter preferred models to those actually available
    choices = []
    for model_id, description, mode in PREFERRED_MODELS:
        if model_id in available:
            choices.append((model_id, description, mode))

    # Also add any other gemma4 models not in preferred list (default to scene)
    for m in available:
        if m.startswith("gemma4") and m not in [c[0] for c in choices]:
            choices.append((m, f"{m}", "scene"))

    if not choices:
        print("No Gemma 4 models found. Available models:")
        for m in available:
            print(f"  {m}")
        print("\nPull a Gemma 4 model: ollama pull gemma4:e4b")
        sys.exit(1)

    if len(choices) == 1:
        return choices[0][0], choices[0][2]

    # Show picker
    if console and HAS_RICH:
        console.print()
        lines = []
        for i, (model_id, desc, mode) in enumerate(choices, 1):
            lines.append(f"  [bold cyan][{i}][/bold cyan] {desc}")
        console.print(Panel(
            "\n".join(lines),
            title="[bold]Select Model[/bold]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        ))
        while True:
            try:
                pick = console.input("[bold yellow]Model>[/bold yellow] ").strip()
                idx = int(pick) - 1
                if 0 <= idx < len(choices):
                    selected = choices[idx]
                    console.print(f"[dim]  → {selected[0]}[/dim]\n")
                    return selected[0], selected[2]
            except (ValueError, KeyboardInterrupt, EOFError):
                pass
            console.print(f"[dim]  Pick 1-{len(choices)}[/dim]")
    else:
        print("\nSelect model:")
        for i, (model_id, desc, mode) in enumerate(choices, 1):
            print(f"  [{i}] {desc}")
        while True:
            try:
                pick = input("Model> ").strip()
                idx = int(pick) - 1
                if 0 <= idx < len(choices):
                    return choices[idx][0], choices[idx][2]
            except (ValueError, KeyboardInterrupt, EOFError):
                pass
            print(f"  Pick 1-{len(choices)}")


def chat(model: str, messages: list, temperature: float = 0.7) -> tuple[str, float]:
    """Send a chat request to Ollama. Returns (response, latency)."""
    start = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    latency = time.time() - start
    return resp.json()["message"]["content"], latency


def generate_suggestions(model: str, conversation: list) -> list[str]:
    """Generate 4 conversation suggestions using the LLM."""
    # Build a summary of the conversation for context
    summary_msgs = [{"role": "system", "content": SUGGESTION_SYSTEM}]

    if len(conversation) > 2:  # At least one exchange happened
        # Extract the last exchange prominently
        last_user = ""
        last_cook = ""
        for msg in reversed(conversation[1:]):
            if msg["role"] == "assistant" and not last_cook:
                last_cook = msg["content"][:120]
            elif msg["role"] == "user" and not last_user:
                last_user = msg["content"][:120]
            if last_user and last_cook:
                break

        # Build earlier context as brief background
        earlier = []
        for msg in conversation[1:-2]:  # Everything before last exchange
            role = "Customer" if msg["role"] == "user" else "Cook"
            earlier.append(f"{role}: {msg['content'][:60]}")
        earlier_text = "\n".join(earlier[-4:])  # Max 2 earlier exchanges

        prompt_parts = []
        if earlier_text:
            prompt_parts.append(f"Earlier:\n{earlier_text}")
        prompt_parts.append(f"JUST NOW — Customer said: \"{last_user}\"")
        prompt_parts.append(f"JUST NOW — Cook responded: \"{last_cook}\"")
        prompt_parts.append("\nWhat would the customer naturally say NEXT? React to what just happened.")

        summary_msgs.append({
            "role": "user",
            "content": "\n".join(prompt_parts),
        })
    elif len(conversation) > 1:
        # First response happened, suggest follow-ups
        last_cook = conversation[-1]["content"][:120] if conversation[-1]["role"] == "assistant" else ""
        summary_msgs.append({
            "role": "user",
            "content": f"The cook just said: \"{last_cook}\"\nWhat would the customer naturally say next?",
        })
    else:
        summary_msgs.append({
            "role": "user",
            "content": "The customer just sat down at the counter. The cook glanced at them. Suggest 4 opening lines.",
        })

    try:
        response, _ = chat(model, summary_msgs, temperature=0.9)
        # Parse JSON array from response
        # Find the JSON array in the response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            suggestions = json.loads(response[start:end])
            if isinstance(suggestions, list) and len(suggestions) >= 4:
                return [str(s) for s in suggestions[:4]]
    except Exception:
        pass

    # Fallback suggestions
    return [
        "What would you recommend?",
        "Busy night tonight?",
        "How long have you been here?",
        "Smells great, what's cooking?",
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Patterns that indicate scene/action text (third-person descriptions)
_SCENE_PATTERNS = [
    re.compile(r"^\*\*scene\*\*", re.IGNORECASE),
    re.compile(r"^\*[^*]"),             # *italics style action*
    re.compile(r"^He\s"),               # He turns, He nods, ...
    re.compile(r"^She\s"),
    re.compile(r"^Kenji\s"),            # Kenji wipes, Kenji looks, ...
    re.compile(r"^The cook\s", re.IGNORECASE),
]


def is_scene_text(text: str) -> bool:
    """Detect if a text segment is a scene description (action, not dialogue)."""
    for pat in _SCENE_PATTERNS:
        if pat.search(text):
            return True
    return False


def clean_scene_markers(text: str) -> str:
    """Remove **scene** markers from text, keep the description."""
    text = re.sub(r"\*\*[Ss]cene\*\*\s*", "", text)
    text = text.strip("*").strip()
    return text


def parse_response(response: str) -> list[tuple[str, str]]:
    """Parse response into segments: ('scene', text) or ('dialogue', text).

    Handles mixed lines like: **scene** He points. "Pork. Egg." **scene** The ajitama.
    """
    segments = []

    for line in response.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Split on **scene** markers to handle inline scene/dialogue mixes
        parts = re.split(r"(\*\*[Ss]cene\*\*)", stripped)

        in_scene = False
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(r"\*\*[Ss]cene\*\*$", part):
                in_scene = True
                continue

            if in_scene:
                # Everything after **scene** until a quote is scene
                # Check if there's quoted dialogue embedded
                quote_match = re.search(r'["“]([^"”]+)["”]', part)
                if quote_match:
                    # Split: scene before quote, dialogue in quote, scene after
                    before = part[:quote_match.start()].strip()
                    dialogue = quote_match.group(0)
                    after = part[quote_match.end():].strip()
                    if before:
                        segments.append(("scene", before))
                    segments.append(("dialogue", dialogue))
                    if after:
                        after_clean = clean_scene_markers(after)
                        if after_clean:
                            segments.append(("scene", after_clean))
                else:
                    segments.append(("scene", part))
                in_scene = False
            elif is_scene_text(part):
                # Scene text detected — but may contain embedded quotes
                clean = clean_scene_markers(part)
                if clean:
                    quote_match = re.search(r'["“]([^"”]+)["”]', clean)
                    if quote_match:
                        before = clean[:quote_match.start()].strip()
                        dialogue = quote_match.group(0)
                        after = clean[quote_match.end():].strip()
                        if before:
                            segments.append(("scene", before))
                        segments.append(("dialogue", dialogue))
                        if after:
                            segments.append(("scene", after))
                    else:
                        segments.append(("scene", clean))
            elif part.startswith('"') or part.startswith('“'):
                segments.append(("dialogue", part))
            else:
                segments.append(("dialogue", part))

    return segments if segments else [("dialogue", response.strip())]


# ---------------------------------------------------------------------------
# Image rendering (welcome banner)
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
INTRO_IMAGE_CANDIDATES = ["intro-alley2.png", "intro_alley.png"]
PERSONA_IMAGE_CANDIDATES = ["persona1.png", "kenji_persona.png"]


def _render_via_term_image(path: Path, target_height: int):
    """Render an image with term-image (better alpha handling) and wrap in Rich Text.

    Uses str(ti) not format(ti) — format() pads the output with terminal-width
    whitespace for centering, which makes Rich think the column is much wider
    than the actual avatar and pushes neighbour cells off-screen.
    """
    if not HAS_TERM_IMAGE:
        return None
    try:
        from rich.text import Text
        ti = BlockImage.from_file(str(path))
        ti.set_size(height=target_height)
        return Text.from_ansi(str(ti))
    except Exception:
        return None


def _render_via_pixels(path: Path, target_height: int):
    """Fallback: render with rich-pixels (Unicode half-blocks, struggles with semi-alpha)."""
    if not HAS_PIXELS:
        return None
    try:
        img = Image.open(path)
        target_pixel_height = target_height * 2  # half-blocks
        aspect = img.width / img.height
        target_pixel_width = int(target_pixel_height * aspect)
        img_resized = img.resize(
            (target_pixel_width, target_pixel_height), Image.LANCZOS
        )
        return Pixels.from_image(img_resized)
    except Exception:
        return None


def _find_asset(candidates: list[str]) -> Path | None:
    for name in candidates:
        path = ASSETS_DIR / name
        if path.exists():
            return path
    return None


def render_intro_image(target_height: int = 15):
    """Return a Rich-renderable for the intro image, or None.

    Prefers term-image (handles transparency cleanly), falls back to rich-pixels.
    """
    path = _find_asset(INTRO_IMAGE_CANDIDATES)
    if path is None:
        return None
    return _render_via_term_image(path, target_height) or _render_via_pixels(path, target_height)


def render_persona_image(target_height: int = 14):
    """Return a Rich-renderable for the Kenji persona avatar, or None."""
    path = _find_asset(PERSONA_IMAGE_CANDIDATES)
    if path is None:
        return None
    return _render_via_term_image(path, target_height) or _render_via_pixels(path, target_height)


# Cache for the per-turn avatar — re-rendering it on every Kenji response would
# be wasteful and the terminal width does not change mid-session.
_PERSONA_CACHE = {"height": None, "render": None, "disabled": False}


def get_persona_render(height: int = 14):
    """Return the cached persona avatar at the given height, or None."""
    if _PERSONA_CACHE["disabled"]:
        return None
    if _PERSONA_CACHE["height"] == height and _PERSONA_CACHE["render"] is not None:
        return _PERSONA_CACHE["render"]
    rendered = render_persona_image(height)
    _PERSONA_CACHE["height"] = height
    _PERSONA_CACHE["render"] = rendered
    return rendered


def disable_persona():
    _PERSONA_CACHE["disabled"] = True
    _PERSONA_CACHE["render"] = None


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------

def format_response_rich(console: "Console", response: str, latency: float):
    """Format Kenji's response with rich styling."""
    segments = parse_response(response)
    formatted = Text()

    for seg_type, text in segments:
        if seg_type == "scene":
            formatted.append(f"  {text}\n", style="italic #6a6a6a")
        else:
            formatted.append(f"  {text}\n", style="bold")

    persona = get_persona_render(14)
    if persona is not None:
        # Avatar left, dialogue right. Larger horizontal padding gives the text
        # breathing room from the avatar, vertical="middle" centers it against
        # the avatar instead of sticking to the top edge.
        body = Table.grid(padding=(0, 4))
        body.add_column(no_wrap=True)
        body.add_column(vertical="middle")
        body.add_row(persona, formatted)
    else:
        body = formatted

    panel = Panel(
        body,
        title="[bold]Kenji[/bold]",
        subtitle=f"[dim]{latency:.1f}s[/dim]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def format_response_plain(response: str, latency: float):
    """Format Kenji's response without rich."""
    segments = parse_response(response)
    print(f"\n--- Kenji ({latency:.1f}s) ---")
    for seg_type, text in segments:
        if seg_type == "scene":
            print(f"  [{text}]")
        else:
            print(f"  {text}")
    print("---\n")


def show_suggestions_rich(console: "Console", suggestions: list[str]):
    """Display conversation suggestions with rich."""
    console.print()
    suggestion_texts = []
    for i, s in enumerate(suggestions, 1):
        suggestion_texts.append(f"  [bold cyan][{i}][/bold cyan] {s}")
    panel = Panel(
        "\n".join(suggestion_texts),
        title="[dim]Suggestions[/dim]",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    console.print(panel)


def show_suggestions_plain(suggestions: list[str]):
    """Display conversation suggestions without rich."""
    print("\nSuggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"  [{i}] {s}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Talk to Kenji at the ramen counter")
    parser.add_argument("--model", default=None,
                        help="Ollama model for Kenji (skip picker)")
    parser.add_argument("--suggestion-model", default=None,
                        help="Ollama model for suggestions (defaults to same as --model)")
    parser.add_argument("--no-suggestions", action="store_true",
                        help="Disable conversation suggestions")
    parser.add_argument("--persona", default="tourist",
                        choices=["tourist", "regular", "stranger"],
                        help="Your character (affects how Kenji sees you)")
    parser.add_argument("--mode", default=None,
                        choices=["scene", "dialogue"],
                        help="Override default mode (scene+dialogue vs dialogue-only)")
    parser.add_argument("--no-image", action="store_true",
                        help="Disable the intro pixel-art image (for terminals that "
                             "render half-blocks incorrectly, e.g. cmder)")
    args = parser.parse_args()

    # Setup console early for picker
    console = Console() if HAS_RICH else None

    # Model selection (also returns recommended mode)
    default_mode = "scene"
    if args.model:
        model = args.model
        if not check_ollama(model):
            sys.exit(1)
        # Look up default mode from registry
        for m_id, _, m_mode in PREFERRED_MODELS:
            if m_id == model:
                default_mode = m_mode
                break
    else:
        model, default_mode = pick_model(console)

    # Mode resolution: explicit flag > model default
    mode = args.mode or default_mode

    suggestion_model = args.suggestion_model or model
    if suggestion_model != model and not check_ollama(suggestion_model):
        sys.exit(1)

    # Load the appropriate spec file for the mode
    spec_filename = SPEC_FILES[mode]
    spec_path = Path(__file__).parent.parent / "characters" / spec_filename
    if not spec_path.exists():
        print(f"Error: Character spec not found at {spec_path}")
        sys.exit(1)
    spec = spec_path.read_text(encoding="utf-8")
    mode_label = "scene+dialogue" if mode == "scene" else "dialogue-only"

    if console:
        mode_style = "[dim]" if mode == "scene" else "[bold yellow]"
        console.print(f"  {mode_style}Mode: {mode_label}[/]  [dim]({spec_filename})[/dim]")

    # Build conversation with system prompt
    conversation = [{"role": "system", "content": spec}]

    # Setup display
    if console:
        console.print()

        if args.no_image:
            disable_persona()
            intro_pixels = None
        else:
            intro_pixels = render_intro_image(target_height=15)
        intro_text = Text()
        intro_text.append("Kenji's Ramen\n", style="bold yellow")
        intro_text.append("Mendokoro Sato — Shinjuku Yokocho\n\n", style="dim")
        intro_text.append("A narrow alley. Steam drifts from under the noren curtain.\n")
        intro_text.append("You push through and sit down at the worn wooden counter.\n")
        intro_text.append("The cook glances at you, then back at his pot.\n\n")
        intro_text.append(f"Model: {model} ({mode_label})\n", style="dim")
        intro_text.append("Type freely or pick a suggestion. ", style="dim")
        intro_text.append("'quit' to leave, 'clear' to start over.\n", style="dim")

        if intro_pixels is not None:
            body = Columns([intro_pixels, intro_text], padding=(0, 2), expand=False)
        else:
            body = intro_text

        console.print(Panel(
            body,
            title="[yellow]麺処 佐藤[/yellow]",
            border_style="yellow",
            box=box.DOUBLE,
            padding=(1, 2),
        ))
    else:
        print("\n" + "=" * 50)
        print("  Kenji's Ramen - Mendokoro Sato")
        print("  Shinjuku Yokocho")
        print("=" * 50)
        print("\nA narrow alley. Steam drifts from under the noren curtain.")
        print("You push through and sit down at the worn wooden counter.")
        print("The cook glances at you, then back at his pot.\n")
        print(f"Model: {model} | Type freely or pick [1-4]")
        print("Type 'quit' to leave, 'clear' to start over\n")

    turn = 0
    suggestions = []

    while True:
        # Generate suggestions
        if not args.no_suggestions:
            if console:
                with console.status("[dim]Thinking of suggestions...[/dim]", spinner="dots"):
                    suggestions = generate_suggestions(suggestion_model, conversation)
                show_suggestions_rich(console, suggestions)
            else:
                suggestions = generate_suggestions(suggestion_model, conversation)
                show_suggestions_plain(suggestions)

        # Get player input
        try:
            if console:
                player_input = console.input("[bold green]You>[/bold green] ")
            else:
                player_input = input("You> ")
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        player_input = player_input.strip()
        if not player_input:
            continue

        if player_input.lower() in ("quit", "exit", "q"):
            if console:
                console.print("\n[dim]You step back through the noren curtain into the alley.[/dim]\n")
            else:
                print("\nYou step back through the noren curtain into the alley.\n")
            break

        if player_input.lower() == "clear":
            conversation = [{"role": "system", "content": spec}]
            turn = 0
            if console:
                console.print("\n[dim]--- New visit ---[/dim]\n")
            else:
                print("\n--- New visit ---\n")
            continue

        # Check if player picked a suggestion number
        if player_input in ("1", "2", "3", "4") and suggestions:
            idx = int(player_input) - 1
            if idx < len(suggestions):
                player_input = suggestions[idx]
                if console:
                    console.print(f"[green]  \"{player_input}\"[/green]")
                else:
                    print(f'  "{player_input}"')

        # Send to Kenji
        conversation.append({"role": "user", "content": player_input})
        turn += 1

        if console:
            with console.status("[yellow]...[/yellow]", spinner="dots"):
                response, latency = chat(model, conversation)
            format_response_rich(console, response, latency)
        else:
            response, latency = chat(model, conversation)
            format_response_plain(response, latency)

        conversation.append({"role": "assistant", "content": response})

        # Show turn counter
        if console:
            console.print(f"[dim]Turn {turn}[/dim]", justify="right")
        else:
            print(f"  [Turn {turn}]")


if __name__ == "__main__":
    main()
