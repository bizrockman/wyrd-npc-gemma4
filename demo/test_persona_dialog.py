"""
Test: persona avatar next to dialog inside one Panel — exact same shape
the real format_response_rich() Panel uses, so we can pick the right
avatar height before wiring it into the conversation render.

Usage:
    python demo/test_persona_dialog.py
"""
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from term_image.image import BlockImage

DEMO_DIR = Path(__file__).resolve().parent
PERSONA_PATH = DEMO_DIR / "assets" / "persona1.png"

# Sample Kenji response — mimics what parse_response() yields:
# dialogue (bold) + scene (italic dim).
SAMPLE_DIALOGUE = '"Comes and goes. Tonight is slow."'
SAMPLE_SCENE = "He keeps wiping the counter, not looking up."


def render_persona(height: int) -> Text:
    ti = BlockImage.from_file(str(PERSONA_PATH))
    ti.set_size(height=height)
    # str(ti) gives just the image; format(ti) pads with terminal-width whitespace
    return Text.from_ansi(str(ti))


def build_response_text() -> Text:
    text = Text()
    text.append(f"  {SAMPLE_DIALOGUE}\n", style="bold")
    text.append(f"  {SAMPLE_SCENE}\n", style="italic #6a6a6a")
    return text


def show_panel(console: Console, height: int):
    """Single panel, two areas: left avatar, right dialogue.
    Matches format_response_rich style exactly (yellow ROUNDED, title,
    subtitle latency).
    """
    persona = render_persona(height)
    text = build_response_text()

    body = Table.grid(padding=(0, 4))
    body.add_column(no_wrap=True)
    body.add_column(vertical="middle")
    body.add_row(persona, text)

    panel = Panel(
        body,
        title="[bold]Kenji[/bold]",
        subtitle=f"[dim]4.2s    avatar={height}[/dim]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def main():
    console = Console()
    console.print()
    for h in (10, 12, 14):
        show_panel(console, h)
        console.print()


if __name__ == "__main__":
    main()
