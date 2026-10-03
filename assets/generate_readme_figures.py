#!/usr/bin/env python3
"""Generate README figures as stable PNG assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / "qwen38_multigpu_20261003"


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


def fitted_font(draw: ImageDraw.ImageDraw, text: str, width: float, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    while size > 12:
        candidate = font(size, bold)
        if draw.textlength(text, font=candidate) <= width:
            return candidate
        size -= 1
    return font(size, bold)


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
    title_font = fitted_font(draw, title, w - 56, 28, bold=True)
    draw.text((x + 28, y + 28), title, fill=color, font=title_font)
    ty = y + 74
    for line in lines:
        draw.text((x + 28, ty), line, fill=COLORS["ink"],
                  font=fitted_font(draw, line, w - 56, 21))
        ty += 34


def overview() -> None:
    """Draw the decode verification loop and mark the two local adapters."""
    img = Image.new("RGB", (1600, 730), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    rounded(draw, (24, 24, 1576, 706), COLORS["bg"], COLORS["line"], radius=30)
    draw.text((80, 74), "DSpark Serving Architecture", fill=COLORS["ink"], font=font(42, True))
    draw.text((80, 130), "Workload-aware verification and shape-aware CUDA Graph dispatch in vLLM",
              fill=COLORS["muted"], font=font(24))

    y, w, h = 235, 320, 190
    cards = [
        (80, "Draft Proposals", ["Matched ~1.99B drafter", "Full 7-token proposals"], COLORS["cyan"], COLORS["cyan_bg"], "UPSTREAM"),
        (450, "Verification Budget", ["Active requests -> K", "Uniform verification prefix"], COLORS["orange"], COLORS["orange_bg"], "LOCAL ADAPTER"),
        (820, "Graph Dispatch", ["query width x requests", "reuse captured shapes"], COLORS["violet"], COLORS["violet_bg"], "LOCAL ADAPTER"),
        (1190, "Target Verification", ["Qwen3.8-27B / TP4", "native rejection + GDN"], COLORS["green"], COLORS["green_bg"], "UPSTREAM"),
    ]
    for x, title, lines, color, fill, ownership in cards:
        draw.text((x + 28, 198), ownership, fill=color, font=font(18, True))
        draw_card(draw, x, y, w, h, title, lines, color, fill)
    for sx in [400, 770, 1140]:
        arrow(draw, (sx, y + h // 2), (sx + 38, y + h // 2))

    draw.line([(1350, 425), (1350, 478), (240, 478), (240, 425)], fill=COLORS["muted"], width=3)
    draw.polygon([(240, 425), (231, 441), (249, 441)], fill=COLORS["muted"])
    centered_text(draw, (370, 492, 1240, 529), "Accepted tokens -> native state update -> next draft round",
                  COLORS["muted"], font(21))

    footer = [
        ("Draft", "fixed 7-token block"),
        ("Target", "query width = K + 1"),
        ("Runtime", "vLLM 0.29 / 4 x A30 / TP4"),
    ]
    for i, (title, value) in enumerate(footer):
        x = 80 + i * 500
        rounded(draw, (x, 580, x + 430, 642), COLORS["card"], COLORS["line"], radius=18)
        title_text = f"{title}: "
        title_font = font(22, True)
        title_width = draw.textlength(title_text, font=title_font)
        value_font = fitted_font(draw, value, 430 - 48 - title_width - 8, 20)
        draw.text((x + 24, 598), title_text, fill=COLORS["ink"], font=title_font)
        draw.text((x + 24 + title_width + 8, 598), value, fill=COLORS["muted"], font=value_font)
    img.save(ROOT / "overview.png")


def speedup() -> None:
    img = Image.new("RGB", (1600, 640), "#ffffff")
    draw = ImageDraw.Draw(img)

    rounded(draw, (24, 24, 1576, 616), "#ffffff", COLORS["line"], width=2, radius=30)
    draw.text((80, 74), "End-to-End Serving Speedup", fill=COLORS["ink"], font=font(42, True))
    draw.text(
        (80, 126),
        "Qwen3 / EAGLE3 checkpoints | c=1 output throughput vs autoregressive decoding",
        fill=COLORS["muted"],
        font=font(24),
    )

    axis_x, axis_y, axis_w = 440, 500, 880
    draw.line((axis_x, axis_y, axis_x + axis_w, axis_y), fill=COLORS["line"], width=3)
    for value in [1.0, 1.2, 1.4, 1.6, 1.8]:
        x = axis_x + int((value - 1.0) / 0.8 * axis_w)
        draw.line((x, 210, x, axis_y), fill="#e2e8f0", width=2)
        centered_text(draw, (x - 40, axis_y + 18, x + 40, axis_y + 52), f"{value:.1f}x", COLORS["muted"], font(20))

    with (ROOT.parent / "results" / "serving_speedup_summary.csv").open(newline="") as handle:
        records = {row["configuration"]: row for row in csv.DictReader(handle)}
    rows = [
        ("Qwen3-8B BF16", "single A30, breakpoint ~c=26",
         float(records["qwen3_8b_bf16_single_a30"]["c1_speedup"]), COLORS["cyan"]),
        ("Qwen3-32B BF16", "TP8, breakpoint ~c=8",
         float(records["qwen3_32b_bf16_tp8"]["c1_speedup"]), COLORS["green"]),
        ("Qwen3-32B INT4", "TP4, breakpoint ~c=5",
         float(records["qwen3_32b_int4_tp4"]["c1_speedup"]), COLORS["orange"]),
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
        "Bar origin: AR = 1.0x. Separate Qwen3 / EAGLE3 study; these values are not Qwen3.8 / DSpark results.",
        fill=COLORS["muted"],
        font=font(21),
    )

    img.save(ROOT / "speedup_summary.png")


def read_results() -> tuple[list[dict], float, dict[str, dict], dict[str, dict]]:
    native = json.loads((RESULTS / "native-paired-resumed.summary.json").read_text())
    budget = json.loads((RESULTS / "budget-v2.summary.json").read_text())
    closure = json.loads((RESULTS / "threeway-resumed.summary.json").read_text())
    if not native["complete"] or not closure["complete"]:
        raise ValueError("Figure inputs must be complete experiments")
    rows = sorted(native["summary"], key=lambda row: row["concurrency"])
    if [row["concurrency"] for row in rows] != [1, 4, 16] or any(row["rounds"] != 5 for row in rows):
        raise ValueError("Expected five paired rounds for c=1/4/16")
    c4 = [row["throughput_ratio"] for row in budget["pairs"]
          if row["concurrency"] == 4 and row["comparison"] == "budget/native"]
    if len(c4) != 5:
        raise ValueError("Expected five held-out budget/native pairs at c=4")
    return (rows, median(c4),
            {row["comparison"]: row for row in closure["comparisons"]},
            {row["mode"]: row for row in closure["modes"]})


def figure_canvas(title: str, subtitle: str, height: float = 7.3):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "text.color": COLORS["ink"], "axes.labelcolor": COLORS["muted"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "axes.titleweight": "bold", "axes.edgecolor": COLORS["line"],
        "savefig.facecolor": COLORS["bg"],
    })
    fig = plt.figure(figsize=(16, height), dpi=100, facecolor=COLORS["bg"])
    fig.add_artist(FancyBboxPatch(
        (0.015, 0.032), 0.97, 0.936, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.02", linewidth=1.3,
        edgecolor=COLORS["line"], facecolor="white", zorder=-10))
    fig.text(0.05, 0.865, title, fontsize=29, fontweight="bold")
    fig.text(0.05, 0.802, subtitle, fontsize=17, color=COLORS["muted"])
    return fig, plt


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#e2e8f0", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=10)


def native_speedup(rows: list[dict]) -> None:
    from matplotlib.ticker import FuncFormatter

    fig, plt = figure_canvas(
        "Qwen3.8-27B | DSpark Serving Speedup",
        "4 x A30 / TP4 | BF16 | ~2K input / 256 output tokens | matched ~1.99B drafter")
    colors = [COLORS["cyan"], COLORS["green"], COLORS["orange"]]
    x = list(range(3))
    ratios = [row["throughput_ratio"] for row in rows]
    labels = [f'c={row["concurrency"]}' for row in rows]
    ax = fig.add_axes((0.075, 0.225, 0.435, 0.455))
    ax.set_title("Output throughput / autoregressive", loc="left", pad=22, fontsize=17)
    ax.bar(x, ratios, color=colors, width=0.52, zorder=3)
    ax.errorbar(x, ratios,
                yerr=[[value - row["throughput_ratio_range"][0] for value, row in zip(ratios, rows)],
                      [row["throughput_ratio_range"][1] - value for value, row in zip(ratios, rows)]],
                fmt="none", ecolor=COLORS["ink"], elinewidth=1.3, capsize=5, zorder=4)
    for pos, value in zip(x, ratios):
        ax.text(pos, value + 0.14, f"{value:.3f}x", ha="center", fontsize=20, fontweight="bold")
    ax.axhline(1.0, color=COLORS["muted"], linestyle=(0, (5, 4)), linewidth=1.4, zorder=2)
    ax.plot([0.69, 0.76], [0.96, 0.96], transform=ax.transAxes,
            color=COLORS["muted"], linestyle=(0, (5, 4)), linewidth=1.4)
    ax.text(0.78, 0.96, "AR = 1.0x", transform=ax.transAxes,
            va="center", fontsize=12, color=COLORS["muted"])
    ax.set_ylim(0, 2.9)
    ax.set_xlim(-0.65, 2.65)
    ax.set_xticks(x, labels, fontsize=16)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}x"))
    clean_axes(ax)

    tp = fig.add_axes((0.625, 0.225, 0.305, 0.455))
    tp.set_title("Client TPOT reduction", loc="left", pad=22, fontsize=17)
    reductions = [row["tpot_reduction_fraction"] * 100 for row in rows]
    tp.barh(x, reductions, height=0.47, color=colors, zorder=3)
    for pos, value in zip(x, reductions):
        tp.text(value + 2, pos, f"-{value:.1f}%", va="center", fontsize=19, fontweight="bold")
    tp.set_yticks(x, labels, fontsize=16)
    tp.set_ylim(2.65, -0.65)
    tp.set_xlim(0, 85)
    tp.set_xticks([0, 20, 40, 60, 80], ["0%", "20%", "40%", "60%", "80%"])
    clean_axes(tp, "x")
    tp.spines["left"].set_visible(False)

    fig.text(0.05, 0.114, "5 paired rounds | median of within-round ratios | whiskers: five-round throughput range",
             fontsize=14, color=COLORS["muted"])
    fig.text(0.05, 0.065, "Independent AR / native deployments. Loading, graph capture and diagnostics excluded from timing.",
             fontsize=12, color=COLORS["muted"])
    fig.savefig(ROOT / "qwen38_speedup.png", dpi=100)
    plt.close(fig)


def framework_optimization(c4: float, comparisons: dict[str, dict], modes: dict[str, dict]) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, plt = figure_canvas(
        "Verification Budget + CUDA Graph Bucketing",
        "vLLM scheduling adaptation | calibrated on separate prompts | frozen policy during validation",
        height=7.6)
    ax = fig.add_axes((0.075, 0.33, 0.405, 0.35))
    ax.set_title("Target graph padding ratio", loc="left", pad=20, fontsize=17)
    padding = [modes[mode]["target_padding_ratio"] for mode in ["native", "graphs", "budget"]]
    ax.bar(range(3), padding, color=[COLORS["orange"], COLORS["green"], COLORS["violet"]], width=0.5, zorder=3)
    for pos, value in enumerate(padding):
        ax.text(pos, value + 0.05, f"{value:.2f}", ha="center", fontsize=22, fontweight="bold")
    ax.set_xticks(range(3), ["Original bins", "Fine bins", "Budget + bins"], fontsize=14)
    ax.set_ylim(0, 1.7)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_ylabel("Captured tokens / effective tokens", fontsize=12)
    clean_axes(ax)

    gain_ax = fig.add_axes((0.61, 0.33, 0.32, 0.35))
    gain_ax.set_title("Throughput gain over original policy", loc="left", pad=20, fontsize=17)
    gains = [(c4 - 1) * 100, (comparisons["budget/native"]["throughput_ratio"] - 1) * 100]
    gain_ax.bar(range(2), gains, color=[COLORS["cyan"], COLORS["violet"]], width=0.45, zorder=3)
    for pos, value in enumerate(gains):
        gain_ax.text(pos, value + 0.4, f"+{value:.1f}%", ha="center", fontsize=23, fontweight="bold")
    gain_ax.set_xticks(range(2), ["c=4", "c=16"], fontsize=16)
    gain_ax.set_ylim(0, 10)
    gain_ax.set_yticks([0, 2, 4, 6, 8, 10], ["0%", "2%", "4%", "6%", "8%", "10%"])
    clean_axes(gain_ax)

    fig.text(0.075, 0.24, "Padding: c=16 shared-pool ablation; shape counts, not GPU-time reduction.",
             fontsize=12, color=COLORS["muted"])
    fig.text(0.075, 0.197, "Gains: c=4 held-out ablation / c=16 direct comparison; five within-round ratio medians.",
             fontsize=12, color=COLORS["muted"])
    fig.add_artist(FancyBboxPatch(
        (0.05, 0.064), 0.90, 0.095, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.018", linewidth=1,
        edgecolor="#bae6fd", facecolor=COLORS["cyan_bg"], zorder=-1))
    ar = comparisons["budget/ar"]
    low, high = ar["throughput_ratio_range"]
    fig.text(0.068, 0.10,
             f'c=16 direct AR check: {ar["throughput_ratio"]:.3f}x throughput  |  range {low:.3f}-{high:.3f}x  |  near parity',
             fontsize=16, fontweight="bold", color=COLORS["cyan"])
    fig.savefig(ROOT / "qwen38_optimization.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    overview()
    speedup()
    native_rows, c4_gain, closure_comparisons, closure_modes = read_results()
    native_speedup(native_rows)
    framework_optimization(c4_gain, closure_comparisons, closure_modes)
    print("Generated overview, Qwen3.8 speedup / optimization, and Qwen3 EAGLE3 charts from recorded results.")
