"""Stage 3 — Chain-of-thought planner.

Backed by Claude Sonnet 4.5 (slide 5). Separates "what to compute" from "how
to express it" — produces a structured plan that the SQL agent and analysis
branches consume. The reasoning trace is captured and surfaced in the UI
(slide 6: "Reasoning trace is captured and surfaced in the UI for inspection").
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import re

from .schemas import DatabaseSchema
from . import config as cfg

@dataclass
class QueryPlan:
    intent: str = ""
    sub_questions: list[str] = field(default_factory=list)
    required_tables: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    grouping: list[str] = field(default_factory=list)
    ordering: str = ""
    complexity: str = "MODERATE"        # SIMPLE / MODERATE / COMPLEX
    chart_hint: str = "bar"
    reasoning: str = ""                  # the full <thinking> trace

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "sub_questions": self.sub_questions,
            "required_tables": self.required_tables,
            "aggregations": self.aggregations,
            "grouping": self.grouping,
            "ordering": self.ordering,
            "complexity": self.complexity,
            "chart_hint": self.chart_hint,
            "reasoning": self.reasoning,
        }

    def get(self, k, default=None):
        return getattr(self, k, default)

# ── Claude prompt ─────────────────────────────────────────────────
PLANNER_SYSTEM = """You are a senior data analyst's planning assistant. Given a question and a database schema, you produce a structured plan BEFORE any SQL is written.

Output format (strict):

<thinking>
Free-form reasoning: identify the intent, decompose into sub-questions, name the required tables and likely joins, the aggregations, the grouping/ordering. Keep this concise — under 200 words.
</thinking>

<plan>
{
  "intent": "<one-sentence statement of what the analyst really wants>",
  "sub_questions": ["..."],
  "required_tables": ["..."],
  "aggregations": ["SUM(...)", "COUNT(*)", ...],
  "grouping": ["..."],
  "ordering": "...",
  "complexity": "SIMPLE|MODERATE|COMPLEX",
  "chart_hint": "bar|line|pie|scatter|area|table"
}
</plan>

Rules:
- Use only tables that appear in the provided schema.
- The plan must be valid JSON inside the <plan> tags.
- The <thinking> block is captured and shown to the user — write it as if explaining to a colleague.
"""

# ════════════════════════════════════════════════════════════════════
# PLANNER
# ════════════════════════════════════════════════════════════════════
class CoTPlanner:
    """Stage 3 — Claude-backed chain-of-thought planner."""

    def __init__(self, anthropic_client, model: str = cfg.CLAUDE_MODEL):
        self.client = anthropic_client
        self.model = model

    def plan(self, question: str, schema: DatabaseSchema,
             route_label: str = "single_chart") -> QueryPlan:
        schema_block = schema.to_prompt_block(max_chars=4000)
        user = (
            f"Task type: {route_label}\n"
            f"\nSCHEMA:\n{schema_block}\n"
            f"\nQUESTION:\n{question}\n"
            f"\nProduce <thinking> and <plan> blocks per the system prompt."
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0.0,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""

        # Extract <thinking> trace
        thinking = ""
        m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
        if m:
            thinking = m.group(1).strip()

        # Extract <plan> JSON
        plan_dict: dict[str, Any] = {}
        m = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
        if m:
            try:
                plan_dict = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                # Try a more forgiving extraction
                inner = m.group(1).strip()
                m2 = re.search(r"\{.*\}", inner, re.DOTALL)
                if m2:
                    try:
                        plan_dict = json.loads(m2.group(0))
                    except json.JSONDecodeError:
                        plan_dict = {}

        return QueryPlan(
            intent=plan_dict.get("intent", question),
            sub_questions=plan_dict.get("sub_questions", []),
            required_tables=plan_dict.get("required_tables", []),
            aggregations=plan_dict.get("aggregations", []),
            grouping=plan_dict.get("grouping", []),
            ordering=plan_dict.get("ordering", ""),
            complexity=plan_dict.get("complexity", "MODERATE"),
            chart_hint=plan_dict.get("chart_hint", "bar"),
            reasoning=thinking or "(planner did not emit <thinking>)",
        )
