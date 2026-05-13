"""Multimodal subsystem — Llama-3.2-Vision extracts a StyleSpec from a reference chart.

Slide 6: "Vision describes; matplotlib executes. Neither hallucinates."
The vision model produces a structured JSON description of palette, grid,
axis style, and color mood — never pixels. The renderer then applies it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re

# ════════════════════════════════════════════════════════════════════
# STYLESPEC
# ════════════════════════════════════════════════════════════════════
@dataclass
class StyleSpec:
    palette: list[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    ])
    bg_color: str = "#FFFFFF"
    fg_color: str = "#000000"
    grid: bool = True
    grid_color: str = "#DDDDDD"
    grid_alpha: float = 0.4
    spine: str = "default"           # 'minimal', 'default', 'all'
    font_family: str = "DejaVu Sans"
    title_size: int = 14
    axis_label_size: int = 11
    color_mood: str = "neutral"      # 'warm', 'cool', 'muted', 'vibrant', 'neutral'
    chart_type_hint: str = "bar"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}

    @classmethod
    def from_dict(cls, d: dict) -> "StyleSpec":
        defaults = cls()
        kwargs = {}
        for k in cls.__annotations__:
            if k in d:
                kwargs[k] = d[k]
            else:
                kwargs[k] = getattr(defaults, k)
        return cls(**kwargs)

# ── Vision prompt ─────────────────────────────────────────────────
STYLE_PROMPT = """Look at this chart image and extract its visual style as JSON.

Return JSON with these keys:
- palette: list of hex colors used in the chart (5-7 colors)
- bg_color: hex of the chart background
- fg_color: hex of the main text color
- grid: true/false (does it show gridlines?)
- grid_color: hex of gridline color (or "#DDDDDD" if no grid)
- grid_alpha: 0.0-1.0 (gridline transparency)
- spine: "minimal" (no top/right spines), "default" (left+bottom only), or "all"
- color_mood: "warm", "cool", "muted", "vibrant", or "neutral"
- chart_type_hint: "bar", "line", "scatter", "pie", "area", or "other"

Reply with JSON ONLY, no commentary, no markdown fences.
"""

SCHEMA_FROM_IMAGE_PROMPT = """Look at this image of database tables / a schema diagram. Extract the schema as JSON:

[
  {
    "name": "table_name",
    "columns": [{"name": "col", "type": "INT64"}, ...]
  },
  ...
]

Reply with JSON ONLY.
"""

# ════════════════════════════════════════════════════════════════════
# VISION MODULE
# ════════════════════════════════════════════════════════════════════
class VisionModule:
    """Llama-3.2-11B-Vision wrapper — slide 6 multimodal card."""

    def __init__(self, llama_vision):
        self.vision = llama_vision

    def extract_style(self, image) -> StyleSpec:
        """Extract a StyleSpec from a reference chart image."""
        raw = self.vision.chat_image(image, STYLE_PROMPT, max_new_tokens=600)

        # Parse JSON (LLMs sometimes wrap in fences or add prose)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                return StyleSpec.from_dict(d)
            except json.JSONDecodeError:
                pass

        print(f"[vision] could not parse StyleSpec from response; using defaults.\n{raw[:300]}")
        return StyleSpec()

    def read_schema(self, image) -> Optional[list[dict]]:
        """Read a schema diagram from an image. Returns list of {name, columns}."""
        raw = self.vision.chat_image(image, SCHEMA_FROM_IMAGE_PROMPT, max_new_tokens=1024)
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None
