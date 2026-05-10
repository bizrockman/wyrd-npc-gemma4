"""Debug: inspect what term-image produces and how Rich measures it."""
from pathlib import Path
from PIL import Image
from term_image.image import BlockImage
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.measure import Measurement
from rich import box

console = Console()
path = Path(__file__).resolve().parent / "assets" / "persona1.png"

img = Image.open(path)
ti = BlockImage(img)
ti.set_size(height=14)
ansi = format(ti)

print(f"persona1: {img.size}, rendered_size: {ti.rendered_size}")
print(f"format() len: {len(ansi)} chars")

# Show line-by-line widths (visible chars only, no ANSI)
import re
ansi_strip = re.compile(r'\x1b\[[0-9;]*m')
lines = ansi.split("\n")
print(f"line count: {len(lines)}")
for i, line in enumerate(lines[:5]):
    visible = ansi_strip.sub("", line)
    print(f"  line {i:2d}: visible_len={len(visible):3d}, raw_len={len(line):4d}")

# What does Rich think the Text width is?
text = Text.from_ansi(ansi)
m = Measurement.get(console, console.options, text)
print(f"\nRich measurement: minimum={m.minimum}, maximum={m.maximum}")

# Same with the dialogue text
dialog = Text()
dialog.append('  "Comes and goes. Tonight is slow."\n', style="bold")
dialog.append("  He keeps wiping the counter, not looking up.\n", style="italic")
m2 = Measurement.get(console, console.options, dialog)
print(f"Dialogue measurement: minimum={m2.minimum}, maximum={m2.maximum}")

# Try the actual layout
print("\n=== Table.grid layout ===")
grid = Table.grid(padding=(0, 2))
grid.add_column(no_wrap=True)
grid.add_column()
grid.add_row(text, dialog)
console.print(Panel(grid, title="Kenji", border_style="yellow", box=box.ROUNDED))
