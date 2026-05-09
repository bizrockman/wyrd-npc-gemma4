"""
Interactive terminal demo for Kenji's Ramen.

Talk to Kenji at the counter. The demo suggests conversation options
based on what a customer might naturally say to a ramen shop owner -
generated independently from Kenji's character to avoid leaking
his internal state.

Usage:
    python kenji_terminal.py
    python kenji_terminal.py --model gemma4:e4b
    python kenji_terminal.py --no-suggestions

Requires: pip install requests rich
"""

import argparse
import json
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
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' for a better terminal experience (pip install rich)")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"

SUGGESTION_SYSTEM = """\
You are a conversation coach helping a customer at a small ramen shop \
in Tokyo. The customer is sitting at the counter. The cook is a quiet, \
middle-aged Japanese man.

Based on the conversation so far, suggest exactly 4 short things the \
customer could naturally say next. Mix types:
- One about the food or cooking
- One casual small-talk / observation about the place
- One slightly more personal question
- One that shows appreciation or curiosity

Rules:
- Keep each suggestion under 15 words
- Sound natural, not scripted
- Don't reference anything the cook hasn't mentioned
- Don't ask about sensitive topics (money, past career, family problems)
- Write as the customer would speak, casual and friendly
- Output ONLY a JSON array of 4 strings, nothing else

Example output:
["How long does the broth take to make?", "Busy night tonight?", \
"You been doing this long?", "This smells amazing, what's in it?"]"""


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def check_ollama(model: str) -> bool:
    """Check if Ollama is running and model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
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

    if len(conversation) > 1:  # More than just the system prompt
        # Give the suggestion model a brief conversation summary
        turns = []
        for msg in conversation[1:]:  # Skip system prompt
            role = "Customer" if msg["role"] == "user" else "Cook"
            turns.append(f"{role}: {msg['content'][:100]}")
        summary = "\n".join(turns[-6:])  # Last 3 exchanges max
        summary_msgs.append({
            "role": "user",
            "content": f"Conversation so far:\n{summary}\n\nSuggest 4 things the customer could say next.",
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
# Terminal UI
# ---------------------------------------------------------------------------

def format_response_rich(console: "Console", response: str, latency: float):
    """Format Kenji's response with rich styling."""
    # Split scene descriptions and dialogue
    lines = response.strip().split("\n")
    formatted = Text()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append("\n")
            continue

        if "**scene**" in stripped.lower() or stripped.startswith("*"):
            # Scene description - italic, dim
            clean = stripped.replace("**scene**", "").replace("**Scene**", "").strip()
            clean = clean.strip("*").strip()
            if clean:
                formatted.append(f"  {clean}\n", style="italic dim")
        elif stripped.startswith('"') or stripped.startswith('“'):
            # Quoted dialogue - bold
            formatted.append(f"  {stripped}\n", style="bold")
        else:
            # Regular text
            formatted.append(f"  {stripped}\n")

    panel = Panel(
        formatted,
        title="[bold]Kenji[/bold]",
        subtitle=f"[dim]{latency:.1f}s[/dim]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def format_response_plain(response: str, latency: float):
    """Format Kenji's response without rich."""
    print(f"\n--- Kenji ({latency:.1f}s) ---")
    print(response)
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
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model for Kenji")
    parser.add_argument("--suggestion-model", default=None,
                        help="Ollama model for suggestions (defaults to same as --model)")
    parser.add_argument("--no-suggestions", action="store_true",
                        help="Disable conversation suggestions")
    parser.add_argument("--persona", default="tourist",
                        choices=["tourist", "regular", "stranger"],
                        help="Your character (affects how Kenji sees you)")
    args = parser.parse_args()

    suggestion_model = args.suggestion_model or args.model

    if not check_ollama(args.model):
        sys.exit(1)
    if suggestion_model != args.model and not check_ollama(suggestion_model):
        sys.exit(1)

    # Load character spec
    spec_path = Path(__file__).parent.parent / "characters" / "kenji_sato.en.yaml"
    if not spec_path.exists():
        print(f"Error: Character spec not found at {spec_path}")
        sys.exit(1)
    spec = spec_path.read_text(encoding="utf-8")

    # Build conversation with system prompt
    conversation = [{"role": "system", "content": spec}]

    # Setup display
    if HAS_RICH:
        console = Console()
        console.print()
        console.print(Panel(
            "[bold]Kenji's Ramen[/bold]\n"
            "[dim]Mendokoro Sato - Shinjuku Yokocho[/dim]\n\n"
            "A narrow alley. Steam drifts from under the noren curtain.\n"
            "You push through and sit down at the worn wooden counter.\n"
            "The cook glances at you, then back at his pot.\n\n"
            f"[dim]Model: {args.model} | Type freely or pick a suggestion[/dim]\n"
            "[dim]Type 'quit' to leave, 'clear' to start over[/dim]",
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
        print(f"Model: {args.model} | Type freely or pick [1-4]")
        print("Type 'quit' to leave, 'clear' to start over\n")

    turn = 0
    suggestions = []

    while True:
        # Generate suggestions
        if not args.no_suggestions:
            if HAS_RICH:
                with console.status("[dim]Thinking of suggestions...[/dim]", spinner="dots"):
                    suggestions = generate_suggestions(suggestion_model, conversation)
                show_suggestions_rich(console, suggestions)
            else:
                suggestions = generate_suggestions(suggestion_model, conversation)
                show_suggestions_plain(suggestions)

        # Get player input
        try:
            if HAS_RICH:
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
            if HAS_RICH:
                console.print("\n[dim]You step back through the noren curtain into the alley.[/dim]\n")
            else:
                print("\nYou step back through the noren curtain into the alley.\n")
            break

        if player_input.lower() == "clear":
            conversation = [{"role": "system", "content": spec}]
            turn = 0
            if HAS_RICH:
                console.print("\n[dim]--- New visit ---[/dim]\n")
            else:
                print("\n--- New visit ---\n")
            continue

        # Check if player picked a suggestion number
        if player_input in ("1", "2", "3", "4") and suggestions:
            idx = int(player_input) - 1
            if idx < len(suggestions):
                player_input = suggestions[idx]
                if HAS_RICH:
                    console.print(f"[green]  \"{player_input}\"[/green]")
                else:
                    print(f'  "{player_input}"')

        # Send to Kenji
        conversation.append({"role": "user", "content": player_input})
        turn += 1

        if HAS_RICH:
            with console.status("[yellow]...[/yellow]", spinner="dots"):
                response, latency = chat(args.model, conversation)
            format_response_rich(console, response, latency)
        else:
            response, latency = chat(args.model, conversation)
            format_response_plain(response, latency)

        conversation.append({"role": "assistant", "content": response})

        # Show turn counter
        if HAS_RICH:
            console.print(f"[dim]Turn {turn}[/dim]", justify="right")
        else:
            print(f"  [Turn {turn}]")


if __name__ == "__main__":
    main()
