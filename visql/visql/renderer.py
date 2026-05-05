"""Chart renderer — matplotlib + StyleSpec.

Slide 6: "matplotlib renders user's data in that style."
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .vision import StyleSpec
from . import config as cfg

# ════════════════════════════════════════════════════════════════════
# STYLE APPLICATION
# ════════════════════════════════════════════════════════════════════
def _apply_style(ax, fig, style: StyleSpec) -> None:
    fig.patch.set_facecolor(style.bg_color)
    ax.set_facecolor(style.bg_color)
    ax.tick_params(colors=style.fg_color, labelsize=style.axis_label_size)
    for sp in ax.spines.values():
        sp.set_color(style.fg_color)
    if style.spine == "minimal":
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    elif style.spine == "default":
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if style.grid:
        ax.grid(True, color=style.grid_color, alpha=style.grid_alpha,
                linestyle="-", linewidth=0.5)
    else:
        ax.grid(False)
    ax.title.set_color(style.fg_color)
    ax.title.set_fontsize(style.title_size)
    ax.xaxis.label.set_color(style.fg_color)
    ax.yaxis.label.set_color(style.fg_color)


def _output_path(name: str) -> Path:
    out = cfg.CACHE_DIR / "charts"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{name}_{int(time.time())}.png"


# ════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINTS
# ════════════════════════════════════════════════════════════════════
def render_chart(df: pd.DataFrame,
                 chart_type: str = "bar",
                 title: str = "",
                 style: Optional[StyleSpec] = None,
                 out_name: str = "chart") -> Optional[str]:
    """Render a single chart to PNG. Returns the file path."""
    if df is None or len(df) == 0:
        return None
    style = style or StyleSpec()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    cols = list(df.columns)
    palette = style.palette

    try:
        if chart_type == "bar" and len(cols) >= 2:
            x = df[cols[0]].astype(str).values
            y = pd.to_numeric(df[cols[1]], errors="coerce").fillna(0).values
            ax.bar(x, y, color=palette[0])
            ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1])
            if len(x) > 6:
                plt.xticks(rotation=30, ha="right")

        elif chart_type == "line" and len(cols) >= 2:
            x = df[cols[0]].values
            for i, c in enumerate(cols[1:5]):
                y = pd.to_numeric(df[c], errors="coerce").fillna(0).values
                ax.plot(x, y, color=palette[i % len(palette)], label=c, linewidth=2)
            ax.set_xlabel(cols[0])
            if len(cols) > 2:
                ax.legend()

        elif chart_type == "scatter" and len(cols) >= 2:
            x = pd.to_numeric(df[cols[0]], errors="coerce").fillna(0).values
            y = pd.to_numeric(df[cols[1]], errors="coerce").fillna(0).values
            ax.scatter(x, y, color=palette[0], alpha=0.6)
            ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1])

        elif chart_type == "pie" and len(cols) >= 2:
            labels = df[cols[0]].astype(str).values
            sizes = pd.to_numeric(df[cols[1]], errors="coerce").fillna(0).values
            ax.pie(sizes, labels=labels, colors=palette[:len(labels)],
                   autopct="%1.1f%%", textprops={"color": style.fg_color})

        elif chart_type == "area" and len(cols) >= 2:
            x = df[cols[0]].values
            y = pd.to_numeric(df[cols[1]], errors="coerce").fillna(0).values
            ax.fill_between(range(len(x)), y, color=palette[0], alpha=0.6)
            ax.set_xticks(range(len(x))); ax.set_xticklabels(x, rotation=30, ha="right")
            ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1])

        else:  # fallback: bar of first numeric column
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                ax.bar(range(len(df)), df[num_cols[0]].values, color=palette[0])
                ax.set_ylabel(num_cols[0])

    except Exception as e:
        ax.text(0.5, 0.5, f"render error: {e}",
                ha="center", va="center", transform=ax.transAxes,
                color=style.fg_color, fontsize=10)

    if title:
        ax.set_title(title)

    _apply_style(ax, fig, style)
    plt.tight_layout()
    out = _output_path(out_name)
    fig.savefig(out, facecolor=style.bg_color, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def render_dashboard(dfs: list[pd.DataFrame],
                     titles: Optional[list[str]] = None,
                     chart_types: Optional[list[str]] = None,
                     style: Optional[StyleSpec] = None,
                     out_name: str = "dashboard") -> Optional[str]:
    """Multi-panel dashboard — slide 5 dashboard route."""
    if not dfs:
        return None
    style = style or StyleSpec()
    titles = titles or [f"Panel {i+1}" for i in range(len(dfs))]
    chart_types = chart_types or ["bar"] * len(dfs)

    n = len(dfs)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), dpi=110)
    fig.patch.set_facecolor(style.bg_color)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes)
    else:
        axes = [a for row in axes for a in row]

    for i, df in enumerate(dfs):
        ax = axes[i]
        ct = chart_types[i] if i < len(chart_types) else "bar"
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        else:
            cols_ = list(df.columns)
            try:
                if ct == "line" and len(cols_) >= 2:
                    ax.plot(df[cols_[0]], df[cols_[1]], color=style.palette[i % len(style.palette)])
                else:
                    x = df[cols_[0]].astype(str).values
                    y = pd.to_numeric(df[cols_[1]], errors="coerce").fillna(0).values if len(cols_) >= 2 else None
                    if y is not None:
                        ax.bar(x, y, color=style.palette[i % len(style.palette)])
                        if len(x) > 6:
                            for lbl in ax.get_xticklabels():
                                lbl.set_rotation(30)
                                lbl.set_ha("right")
            except Exception as e:
                ax.text(0.5, 0.5, f"err: {e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
        ax.set_title(titles[i])
        _apply_style(ax, fig, style)

    # hide any unused axes
    for j in range(len(dfs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = _output_path(out_name)
    fig.savefig(out, facecolor=style.bg_color, bbox_inches="tight")
    plt.close(fig)
    return str(out)
