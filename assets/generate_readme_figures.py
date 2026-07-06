#!/usr/bin/env python3
"""Generate README figures as stable PNG assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


COLORS = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#cbd5e1",
    "cyan": "#0891b2",
    "cyan_bg": "#ecfeff",
    "green": "#16a34a",
    "green_bg": "#f0fdf4",
    "orange": "#ea580c",
    "orange_bg": "#fff7ed",
    "violet": "#7c3aed",
    "violet_bg": "#f5f3ff",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, width: int = 2, radius: int = 24) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str, fnt: ImageFont.FreeTypeFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) / 2 - 2
    draw.text((x, y), text, fill=fill, font=fnt)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = COLORS["muted"]) -> None:
    draw.line([start, end], fill=color, width=4)
    x, y = end
    draw.polygon([(x, y), (x - 16, y - 9), (x - 16, y + 9)], fill=color)


def draw_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, lines: list[str], color: str, fill: str) -> None:
    rounded(draw, (x, y, x + w, y + h), fill, color, width=3, radius=24)
    draw.text((x + 28, y + 28), title, fill=color, font=font(28, True))
    ty = y + 74
    for line in lines:
        draw.text((x + 28, ty), line, fill=COLORS["ink"], font=font(21))
        ty += 34


def overview() -> None:
    img = Image.new("RGB", (1600, 620), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    rounded(draw, (24, 24, 1576, 596), COLORS["bg"], COLORS["line"], width=2, radius=30)
    draw.text((80, 74), "DSpark Speculative Decoding Serving Benchmark", fill=COLORS["ink"], font=font(42, True))
    draw.text(
        (80, 126),
        "From accepted-length reproduction to OpenAI-compatible serving speedup",
        fill=COLORS["muted"],
        font=font(24),
    )

    y, w, h = 220, 320, 170
    cards = [
        (80, "DeepSpec Eval", ["DSpark / EAGLE3 / DFlash", "Qwen3-8B acceptance"], COLORS["cyan"], COLORS["cyan_bg"]),
        (450, "Algorithm Signal", ["accepted length", "paper-level reproduction"], COLORS["green"], COLORS["green_bg"]),
        (820, "Serving A/B", ["target-only baseline", "target + draft + verify"], COLORS["orange"], COLORS["orange_bg"]),
        (1190, "Adaptive Policy", ["fit breakpoint", "route or fallback"], COLORS["violet"], COLORS["violet_bg"]),
    ]
    for x, title, lines, color, fill in cards:
        draw_card(draw, x, y, w, h, title, lines, color, fill)

    for sx in [400, 770, 1140]:
        arrow(draw, (sx, y + h // 2), (sx + 38, y + h // 2))

    metrics = [
        ("Metrics", "TTFT / TPOT / P95 / tokens/s"),
        ("Stack", "Qwen3 + vLLM/SGLang + A30"),
        ("Artifacts", "policy + CSVs + reports"),
    ]
    for i, (title, body) in enumerate(metrics):
        x = 80 + i * 500
        rounded(draw, (x, 470, x + 430, 532), COLORS["card"], COLORS["line"], width=2, radius=18)
        title_text = f"{title}: "
        title_font = font(22, True)
        body_font = font(20)
        draw.text((x + 24, 488), title_text, fill=COLORS["ink"], font=title_font)
        title_width = draw.textbbox((0, 0), title_text, font=title_font)[2]
        draw.text((x + 24 + title_width + 8, 488), body, fill=COLORS["muted"], font=body_font)

    img.save(ROOT / "overview.png")


def speedup() -> None:
    img = Image.new("RGB", (1600, 640), "#ffffff")
    draw = ImageDraw.Draw(img)

    rounded(draw, (24, 24, 1576, 616), "#ffffff", COLORS["line"], width=2, radius=30)
    draw.text((80, 74), "End-to-End Serving Speedup", fill=COLORS["ink"], font=font(42, True))
    draw.text(
        (80, 126),
        "Measured through OpenAI-compatible baseline/speculative serving endpoints",
        fill=COLORS["muted"],
        font=font(24),
    )

    axis_x, axis_y, axis_w = 380, 500, 940
    draw.line((axis_x, axis_y, axis_x + axis_w, axis_y), fill=COLORS["line"], width=3)
    for value in [1.0, 1.2, 1.4, 1.6, 1.8]:
        x = axis_x + int((value - 1.0) / 0.8 * axis_w)
        draw.line((x, 210, x, axis_y), fill="#e2e8f0", width=2)
        centered_text(draw, (x - 40, axis_y + 18, x + 40, axis_y + 52), f"{value:.1f}x", COLORS["muted"], font(20))

    rows = [
        ("Qwen3-8B BF16", "single A30, breakpoint ~c=26", 1.76, COLORS["cyan"]),
        ("Qwen3-32B BF16", "TP8, breakpoint ~c=8", 1.57, COLORS["green"]),
        ("Qwen3-32B INT4", "TP4, breakpoint ~c=5", 1.43, COLORS["orange"]),
    ]
    for i, (name, detail, value, color) in enumerate(rows):
        y = 226 + i * 92
        draw.text((80, y), name, fill=COLORS["ink"], font=font(26, True))
        draw.text((80, y + 34), detail, fill=COLORS["muted"], font=font(20))
        bar_w = int((value - 1.0) / 0.8 * axis_w)
        draw.rounded_rectangle((axis_x, y + 4, axis_x + bar_w, y + 48), radius=14, fill=color)
        draw.text((axis_x + bar_w + 24, y + 10), f"{value:.2f}x", fill=COLORS["ink"], font=font(26, True))

    draw.text(
        (80, 572),
        "Useful region: low-to-moderate concurrency, long outputs, matched workload, and decode-bound serving.",
        fill=COLORS["muted"],
        font=font(21),
    )

    img.save(ROOT / "speedup_summary.png")


if __name__ == "__main__":
    overview()
    speedup()
