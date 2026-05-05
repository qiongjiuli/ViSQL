"""Style imitation evaluation — CIELAB ΔE-76 self-consistency.

Slide 10: ΔE 7.4 mean across 12 reference charts; chart-type accuracy 0.83.
Below the JND threshold of 10 for perceptible color difference.

Self-consistency loop:
    StyleSpec₁ = vision.extract_style(ref_img)
    rendered  = render_chart(seed_data, style=StyleSpec₁)
    StyleSpec₂ = vision.extract_style(rendered)
    ΔE        = mean_color_distance(StyleSpec₁.palette, StyleSpec₂.palette)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from visql.vision   import VisionModule, StyleSpec
from visql.renderer import render_chart


# ════════════════════════════════════════════════════════════════════
# CIELAB DISTANCE
# ════════════════════════════════════════════════════════════════════
def hex_to_lab(hexcolor: str) -> np.ndarray:
    """Convert #RRGGBB hex to CIELAB. Uses skimage if available, else manual sRGB→XYZ→Lab."""
    try:
        from skimage.color import rgb2lab
        h = hexcolor.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        rgb = np.array([[[r / 255.0, g / 255.0, b / 255.0]]])
        return rgb2lab(rgb)[0, 0]
    except ImportError:
        return _manual_hex_to_lab(hexcolor)


def _manual_hex_to_lab(hexcolor: str) -> np.ndarray:
    """Manual sRGB → XYZ (D65) → Lab. Used if skimage isn't available."""
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    # gamma decompand
    def _ungamma(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    R, G, B = _ungamma(r), _ungamma(g), _ungamma(b)
    # sRGB → XYZ
    X = 0.4124 * R + 0.3576 * G + 0.1805 * B
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Z = 0.0193 * R + 0.1192 * G + 0.9505 * B
    # XYZ → Lab (D65)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return np.array([L, a, b_])


def palette_delta_e(p1: list[str], p2: list[str]) -> float:
    """Mean ΔE-76 between two palettes after pairing the closest colors."""
    if not p1 or not p2:
        return float("nan")
    lab1 = np.array([hex_to_lab(c) for c in p1])
    lab2 = np.array([hex_to_lab(c) for c in p2])
    # For each color in p1, find the closest in p2; average the distances.
    dists = []
    for v1 in lab1:
        d = np.linalg.norm(lab2 - v1, axis=1)
        dists.append(d.min())
    return float(np.mean(dists))


# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
@dataclass
class StyleEvalSummary:
    n: int = 0
    mean_delta_e: float = 0.0
    median_delta_e: float = 0.0
    chart_type_accuracy: float = 0.0
    per_image: list = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# EVALUATOR
# ════════════════════════════════════════════════════════════════════
class StyleEvaluator:
    def __init__(self, vision: VisionModule):
        self.vision = vision

    def evaluate(self, ref_image_paths: list[str | Path],
                 seed_df: pd.DataFrame = None,
                 verbose: bool = False) -> StyleEvalSummary:
        if seed_df is None:
            seed_df = pd.DataFrame({
                "category": list("ABCDEFG"),
                "value": [120, 95, 80, 65, 55, 42, 30],
            })

        per_image = []
        delta_es = []
        type_correct = 0

        for path in ref_image_paths:
            try:
                style1 = self.vision.extract_style(str(path))
                rendered = render_chart(seed_df, chart_type=style1.chart_type_hint or "bar",
                                         style=style1, out_name=f"styleeval_{Path(path).stem}")
                if rendered is None:
                    continue
                style2 = self.vision.extract_style(rendered)
                de = palette_delta_e(style1.palette, style2.palette)
                delta_es.append(de)
                if style1.chart_type_hint == style2.chart_type_hint:
                    type_correct += 1
                per_image.append({
                    "ref": str(path),
                    "delta_e": de,
                    "type1": style1.chart_type_hint,
                    "type2": style2.chart_type_hint,
                    "rendered": rendered,
                })
                if verbose:
                    print(f"  {Path(path).name}  ΔE={de:.2f}  type {style1.chart_type_hint}→{style2.chart_type_hint}")
            except Exception as e:
                print(f"[style_eval] error on {path}: {e}")

        n = len(per_image)
        return StyleEvalSummary(
            n=n,
            mean_delta_e=float(np.mean(delta_es)) if delta_es else float("nan"),
            median_delta_e=float(np.median(delta_es)) if delta_es else float("nan"),
            chart_type_accuracy=type_correct / n if n else 0.0,
            per_image=per_image,
        )
