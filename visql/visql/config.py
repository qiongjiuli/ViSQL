"""Centralized configuration for ViSQL v2."""
from __future__ import annotations
import os
from pathlib import Path
import torch

# ── Models ─────────────────────────────────────────────────────────
LLAMA_TEXT_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
SQL_BASE_MODEL     = "defog/sqlcoder-7b-2"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL       = "claude-sonnet-4-5-20250929"

# ── Hardware ───────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ── Paths ──────────────────────────────────────────────────────────
ROOT_DIR  = Path(os.environ.get("VISQL_ROOT", "/content/visql_v2"))
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR  = ROOT_DIR / "data"
LORA_DIR  = ROOT_DIR / "lora_adapters"
for d in (CACHE_DIR, DATA_DIR, LORA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Generation hyperparameters ────────────────────────────────────
TEXT_MAX_NEW_TOKENS = 768
TEXT_TEMPERATURE    = 0.0      # router/planner/report — deterministic
SQL_MAX_NEW_TOKENS  = 512
SQL_TEMPERATURE     = 0.0
SQL_MAX_RETRIES     = 3        # self-correct loop (slide 7)

# ── Task labels (router output space) ─────────────────────────────
TASK_LABELS = ["single_chart", "dashboard", "ab_test", "ml_modeling", "sql_only"]

# ── A/B gate keywords (slide 4: strict gate) ──────────────────────
EXPERIMENT_KEYWORDS = (
    "a/b test", "ab test", "a-b test", "experiment", "treatment", "control",
    "variant", "holdout", "randomized", "lift", "uplift", "intent-to-treat",
    "checkout_redesign", "feature flag",
)

# ── Retrieval ──────────────────────────────────────────────────────
TOP_K_TABLES   = 5
TOP_K_EXEMPLARS = 3

# ── Vision ─────────────────────────────────────────────────────────
VISION_MAX_NEW_TOKENS = 1024
