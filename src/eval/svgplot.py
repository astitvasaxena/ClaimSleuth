"""
Minimal, dependency-free SVG charts for the evaluation (matplotlib is not installed and
was not added). Boxplots use the standard definition: box = Q1..Q3 with the median
line, whiskers to the most extreme points within 1.5 x IQR, outliers drawn as dots.
Heat-map cells with no measurement are drawn grey and labelled "n/a" (never as zero).
"""
from html import escape
from typing import Dict, List, Optional, Sequence


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (same as numpy's default)."""
    if not sorted_values:
        raise ValueError("no values")
    pos = (len(sorted_values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def box_stats(values: Sequence[float]) -> Dict[str, float]:
    v = sorted(values)
    q1, med, q3 = quantile(v, 0.25), quantile(v, 0.5), quantile(v, 0.75)
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = [x for x in v if lo_fence <= x <= hi_fence]
    return {"n": len(v), "q1": q1, "median": med, "q3": q3,
            "whisker_low": min(inside), "whisker_high": max(inside),
            "outliers": [x for x in v if x < lo_fence or x > hi_fence]}


def _ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / count
    return [lo + i * step for i in range(count + 1)]


def boxplot_svg(series: Dict[str, List[float]], title: str, ylabel: str, width: int = 760, height: int = 420) -> str:
    series = {k: v for k, v in series.items() if v}
    left, right, top, bottom = 70, 20, 48, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [x for v in series.values() for x in v]
    lo, hi = min(all_values), max(all_values)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = max(0.0, lo - pad), hi + pad
    y = lambda val: top + plot_h - (val - lo) / (hi - lo) * plot_h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="15" font-weight="bold">{escape(title)}</text>',
           f'<text x="16" y="{top + plot_h / 2}" transform="rotate(-90 16 {top + plot_h / 2})" text-anchor="middle">{escape(ylabel)}</text>']
    for t in _ticks(lo, hi):
        out.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.0f}</text>')
    slot = plot_w / max(1, len(series))
    for i, (label, values) in enumerate(series.items()):
        s = box_stats(values)
        cx, bw = left + slot * (i + 0.5), min(60, slot * 0.5)
        out.append(f'<line x1="{cx}" x2="{cx}" y1="{y(s["whisker_low"]):.1f}" y2="{y(s["q1"]):.1f}" stroke="#333"/>')
        out.append(f'<line x1="{cx}" x2="{cx}" y1="{y(s["q3"]):.1f}" y2="{y(s["whisker_high"]):.1f}" stroke="#333"/>')
        for w in (s["whisker_low"], s["whisker_high"]):
            out.append(f'<line x1="{cx - bw / 4}" x2="{cx + bw / 4}" y1="{y(w):.1f}" y2="{y(w):.1f}" stroke="#333"/>')
        out.append(f'<rect x="{cx - bw / 2}" y="{y(s["q3"]):.1f}" width="{bw}" height="{max(1, y(s["q1"]) - y(s["q3"])):.1f}" '
                   f'fill="#9ecae1" stroke="#2c5d87"/>')
        out.append(f'<line x1="{cx - bw / 2}" x2="{cx + bw / 2}" y1="{y(s["median"]):.1f}" y2="{y(s["median"]):.1f}" '
                   f'stroke="#08306b" stroke-width="2"/>')
        for o in s["outliers"]:
            out.append(f'<circle cx="{cx}" cy="{y(o):.1f}" r="2.5" fill="none" stroke="#c0392b"/>')
        out.append(f'<text x="{cx}" y="{top + plot_h + 18}" text-anchor="middle">{escape(label)}</text>')
        out.append(f'<text x="{cx}" y="{top + plot_h + 34}" text-anchor="middle" fill="#666">n={s["n"]}, '
                   f'med={s["median"]:.0f}</text>')
    out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#999"/>')
    out.append("</svg>")
    return "\n".join(out)


def heatmap_svg(rows: List[str], cols: List[str], values: List[List[Optional[float]]], title: str,
                fmt: str = "{:.1f}", normalise: str = "column", cell_w: int = 120, cell_h: int = 28) -> str:
    """normalise='column' scales colours per column (metrics with different units); 'global' uses one scale."""
    left, top = 170, 60
    width, height = left + cell_w * len(cols) + 20, top + cell_h * len(rows) + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">{escape(title)}</text>']
    for j, col in enumerate(cols):
        out.append(f'<text x="{left + cell_w * (j + 0.5)}" y="{top - 10}" text-anchor="middle">{escape(col)}</text>')

    def scale(j):
        vals = ([values[i][j] for i in range(len(rows))] if normalise == "column"
                else [v for row in values for v in row])
        vals = [v for v in vals if v is not None]
        return (min(vals), max(vals)) if vals else (0.0, 1.0)

    for i, row in enumerate(rows):
        out.append(f'<text x="{left - 8}" y="{top + cell_h * (i + 0.5) + 4}" text-anchor="end">{escape(row)}</text>')
        for j in range(len(cols)):
            v = values[i][j]
            x, yy = left + cell_w * j, top + cell_h * i
            if v is None:
                out.append(f'<rect x="{x}" y="{yy}" width="{cell_w}" height="{cell_h}" fill="#dddddd" stroke="white"/>')
                out.append(f'<text x="{x + cell_w / 2}" y="{yy + cell_h / 2 + 4}" text-anchor="middle" fill="#777">n/a</text>')
                continue
            lo, hi = scale(j)
            t = 0.0 if hi == lo else (v - lo) / (hi - lo)
            r, g, b = int(255 - 47 * t), int(255 - 160 * t), int(255 - 200 * t)
            text = "#fff" if t > 0.6 else "#111"
            out.append(f'<rect x="{x}" y="{yy}" width="{cell_w}" height="{cell_h}" fill="rgb({r},{g},{b})" stroke="white"/>')
            out.append(f'<text x="{x + cell_w / 2}" y="{yy + cell_h / 2 + 4}" text-anchor="middle" fill="{text}">'
                       f'{escape(fmt.format(v))}</text>')
    out.append("</svg>")
    return "\n".join(out)
