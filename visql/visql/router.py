"""Stage 1 — Task router.

5-way classifier (single_chart / dashboard / ab_test / ml_modeling / sql_only)
implemented as a Llama-3.1-8B JSON-output prompt, plus a STRICT A/B test gate
that re-routes observational comparisons to single_chart.

The gate is the architectural decision featured on slide 4 and demoed on
slide 9. It is intentionally redundant with the LLM: even if Llama proposes
ab_test, the gate refuses unless:
    (a) the question contains experiment-language keywords, OR
    (b) the schema has a column suggesting variant assignment.
This prevents the report writer from making causal claims it can't support.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import re

from . import config as cfg
from .schemas import DatabaseSchema

@dataclass
class RouterDecision:
    label: str                 # one of cfg.TASK_LABELS
    confidence: float
    rationale: str
    gated_from: Optional[str] = None   # if the gate demoted, original label here
    raw_response: str = ""

    def get(self, k, default=None):
        """Dict-like access for UI convenience."""
        return getattr(self, k, default)

# ── Router system prompt ──────────────────────────────────────────
ROUTER_SYSTEM = """You are a task router for a data analyst assistant. Classify the user's question into ONE of these task types:

- single_chart   : one focused visualization answering a specific question
- dashboard      : multiple coordinated charts giving an overview
- ab_test        : statistical analysis of a randomized experiment with explicit treatment vs control
- ml_modeling    : fit a predictive model (classification, regression, etc.)
- sql_only       : data extraction with no visualization or analysis needed

CRITICAL distinction for ab_test:
- ab_test ONLY applies when the question is about a real randomized experiment.
- Comparisons between observed segments (mobile vs desktop, US vs UK, etc.) are NOT a/b tests — those are observational and route to single_chart.
- A/B test indicators: "experiment", "treatment", "control", "variant", "lift", "uplift", "randomized".

Reply in valid JSON only:
{"label": "<label>", "confidence": <0.0-1.0>, "rationale": "<brief reason>"}
"""

# ════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════
class TaskRouter:
    """Stage 1 — task classifier + strict A/B test gate."""

    def __init__(self, llama_text):
        self.llm = llama_text

    def _llama_classify(self, question: str, schema: DatabaseSchema) -> RouterDecision:
        """Ask Llama to classify. Returns parsed RouterDecision (pre-gate)."""
        schema_brief = f"Database: {schema.db_id}, tables: " + ", ".join(
            t.name for t in schema.tables[:8]
        )
        user_prompt = f"{schema_brief}\n\nQuestion: {question}\n\nReply with JSON only."

        raw = self.llm.chat(ROUTER_SYSTEM, user_prompt,
                            max_new_tokens=200, temperature=0.0)

        # Try to parse JSON anywhere in the response
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                label = parsed.get("label", "single_chart").strip()
                if label not in cfg.TASK_LABELS:
                    label = "single_chart"
                return RouterDecision(
                    label=label,
                    confidence=float(parsed.get("confidence", 0.7)),
                    rationale=parsed.get("rationale", "(none)"),
                    raw_response=raw,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: keyword heuristics
        return self._heuristic_route(question, raw)

    @staticmethod
    def _heuristic_route(question: str, raw: str = "") -> RouterDecision:
        """Backup classifier when JSON parsing fails."""
        q = question.lower()
        if any(k in q for k in cfg.EXPERIMENT_KEYWORDS):
            return RouterDecision("ab_test", 0.7, "(heuristic) experiment keyword", raw_response=raw)
        if any(k in q for k in ("predict", "classify", "forecast", "model the", "build a model")):
            return RouterDecision("ml_modeling", 0.7, "(heuristic) predictive language", raw_response=raw)
        if any(k in q for k in ("dashboard", "overview", "summary across", "kpis")):
            return RouterDecision("dashboard", 0.6, "(heuristic) dashboard language", raw_response=raw)
        if any(k in q for k in ("just give me", "extract", "list", "raw data")):
            return RouterDecision("sql_only", 0.6, "(heuristic) extraction language", raw_response=raw)
        return RouterDecision("single_chart", 0.65, "(heuristic) default", raw_response=raw)

    @staticmethod
    def question_has_experiment_language(question: str) -> bool:
        q = question.lower()
        return any(k in q for k in cfg.EXPERIMENT_KEYWORDS)

    def apply_gate(self, decision: RouterDecision,
                   question: str, schema: DatabaseSchema) -> RouterDecision:
        """STRICT A/B gate (slide 4).

        If Llama proposed `ab_test`, only allow it if:
          - question contains experiment-language, OR
          - schema has a variant-like column.
        Otherwise demote to `single_chart`.
        """
        if decision.label != "ab_test":
            return decision

        has_keywords = self.question_has_experiment_language(question)
        has_variant_col = schema.has_variant_column()

        if has_keywords or has_variant_col:
            return decision  # passes the gate

        gated = RouterDecision(
            label="single_chart",
            confidence=decision.confidence,
            rationale=(
                "Gated from ab_test: no experiment language in the question "
                "and no variant column in the schema. Treating as observational "
                "comparison; routing to single_chart to avoid unsupported causal claims."
            ),
            gated_from="ab_test",
            raw_response=decision.raw_response,
        )
        return gated

    def route(self, question: str, schema: DatabaseSchema) -> RouterDecision:
        """Stage 1 entrypoint. Llama classifies, then the strict gate runs."""
        # First-pass: do we even need Llama? If question explicitly mentions
        # an experiment AND schema has a variant column, this is unambiguous.
        if (self.question_has_experiment_language(question)
                and schema.has_variant_column()):
            return RouterDecision(
                label="ab_test", confidence=0.95,
                rationale="Question contains experiment-language and schema has variant column.",
            )

        decision = self._llama_classify(question, schema)
        return self.apply_gate(decision, question, schema)
