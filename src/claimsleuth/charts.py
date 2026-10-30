"""Charts drawn from measured rows: the per-run agent heat map (dashboard) and the evaluation charts (eval/)."""
import io
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from claimsleuth import db  # noqa: E402
from claimsleuth.callbacks.metrics import METRIC_COLUMNS  # noqa: E402

LABELS = {"seconds": "seconds", "prompt_tokens": "prompt tokens", "completion_tokens": "completion tokens",
          "llm_calls": "model calls", "tool_calls": "tool calls", "errors": "errors", "guardrail_hits": "guardrail hits"}


def _png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def heatmap(rows: Sequence[dict], title: str, columns: Sequence[str] = METRIC_COLUMNS) -> bytes:
    """Agents down the side, metrics across. Each column is scaled 0-1 on its own; cells show the raw values."""
    agents = [r["agent"] for r in rows]
    raw = np.array([[float(r.get(c) or 0) for c in columns] for r in rows]) if rows else np.zeros((0, len(columns)))
    peak = raw.max(axis=0) if len(raw) else np.ones(len(columns))
    scaled = np.divide(raw, peak, out=np.zeros_like(raw), where=peak > 0)
    fig, ax = plt.subplots(figsize=(1.25 * len(columns) + 2.2, 0.48 * max(len(agents), 1) + 1.4))
    ax.imshow(scaled, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), [LABELS.get(c, c) for c in columns], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(agents)), agents, fontsize=9)
    for i in range(len(agents)):
        for j, column in enumerate(columns):
            value = raw[i, j]
            text = f"{value:.1f}" if column == "seconds" else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if scaled[i, j] > 0.6 else "black")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(agents)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    return _png(fig)


def run_heatmap(run_id: str) -> Optional[bytes]:
    with db.connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        rows = [dict(r) for r in conn.execute("SELECT * FROM agent_metrics WHERE run_id = ? ORDER BY agent", (run_id,))]
    if run is None or not rows:
        return None
    return heatmap(rows, f"Run {run_id}: {run['claim_id']}, {run['pattern']} / {run['prompt_style']}")


def boxplot(groups: dict[str, list[float]], ylabel: str, title: str) -> bytes:
    labels = list(groups)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(labels) + 2), 4))
    ax.boxplot([groups[k] for k in labels], tick_labels=labels, showmeans=True)
    for i, k in enumerate(labels, start=1):
        ax.scatter(np.full(len(groups[k]), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(groups[k])),
                   groups[k], s=10, alpha=0.5, color="tab:blue")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.3)
    return _png(fig)
