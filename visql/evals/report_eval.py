"""LLM-as-judge report evaluation — 4-axis rubric, Claude judges.

Slide 10: 4.1/5 mean overall, coverage 0.78. Axes: factual / complete /
clarity / actionable, each 1-5. Hard guard prevents fabrication on empty
data — verified across all 6 reference cases.

Slide 11: cross-family rationale. Same-family judging inflates scores. Our
setup uses Claude as judge — cross-family on the Llama-output side, with
the trade-off that the Claude-output side is technically same-family. We
document this honestly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import re

from visql import config as cfg

# ════════════════════════════════════════════════════════════════════
# REFERENCE QUESTIONS (the 6 in the deck)
# ════════════════════════════════════════════════════════════════════
REFERENCE_QUESTIONS = [
    {
        "id": "thelook_q1",
        "db_id": "thelook",
        "question": "Show me the top 10 product categories by revenue.",
        "expected_findings": [
            "names the top category",
            "cites at least one revenue figure",
            "shows top 10 in some form",
        ],
    },
    {
        "id": "thelook_q2",
        "db_id": "thelook",
        "question": "What's the return rate by product category?",
        "expected_findings": [
            "highlights highest-return category",
            "cites a return-rate percentage",
        ],
    },
    {
        "id": "thelook_q3",
        "db_id": "thelook",
        "question": "How does monthly revenue trend over the past 12 months?",
        "expected_findings": [
            "describes the trend direction",
            "cites at least one month value",
            "names a peak or trough month",
        ],
    },
    {
        "id": "thelook_q4_ab",
        "db_id": "thelook",
        "question": "For the checkout_redesign_2024 A/B test, did treatment lift conversion vs control?",
        "expected_findings": [
            "cites lift in percentage points",
            "cites a p-value or significance",
            "states whether result is significant",
        ],
    },
    {
        "id": "sec_q1",
        "db_id": "sec",
        "question": "Top 5 companies by total revenue in the most recent quarter.",
        "expected_findings": [
            "names the top company",
            "cites at least one revenue figure",
        ],
    },
    {
        "id": "thelook_q_empty",
        "db_id": "thelook",
        "question": "Show orders from year 1900 by user country.",
        "expected_findings": [
            "states no data was returned",
            "does not fabricate findings",
        ],
        "expect_empty": True,
    },
]

# ════════════════════════════════════════════════════════════════════
# JUDGE PROMPT
# ════════════════════════════════════════════════════════════════════
JUDGE_SYSTEM = """You are an expert data-analysis report judge. Score the report on 4 axes (each 1-5):

  - factual_accuracy : Are cited numbers / facts plausible given the evidence?
  - completeness    : Does the report cover the expected findings?
  - clarity         : Is the writing clear and well-structured?
  - actionability   : Does it state implications or next steps?

Also produce coverage: a list of booleans, one per expected finding, indicating whether each was addressed.

Reply in valid JSON only:
{
  "factual_accuracy": <int 1-5>,
  "completeness":     <int 1-5>,
  "clarity":          <int 1-5>,
  "actionability":    <int 1-5>,
  "coverage":         [<true|false>, ...],
  "overall_comment":  "<brief one-sentence summary>"
}
"""

# ════════════════════════════════════════════════════════════════════
# DATACLASSES
# ════════════════════════════════════════════════════════════════════
@dataclass
class ReportJudgement:
    question_id: str
    factual_accuracy: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    actionability: float = 0.0
    overall: float = 0.0
    coverage: list = field(default_factory=list)
    coverage_rate: float = 0.0
    overall_comment: str = ""
    expected_empty: bool = False
    actually_empty_handled: bool = False


# ════════════════════════════════════════════════════════════════════
# JUDGE
# ════════════════════════════════════════════════════════════════════
class ReportJudge:
    """Cross-family LLM-as-judge — Claude scores reports."""

    def __init__(self, anthropic_client, model: str = cfg.CLAUDE_MODEL):
        self.client = anthropic_client
        self.model = model

    def judge_one(self, ref: dict, report: str, evidence_summary: str = "") -> ReportJudgement:
        # Empty-data special case
        expected_empty = bool(ref.get("expect_empty", False))
        empty_phrases = ("no data", "0 rows", "zero rows", "did not execute",
                         "declining to write", "returned 0")
        empty_handled = any(p in report.lower() for p in empty_phrases)

        user = (
            f"QUESTION: {ref['question']}\n\n"
            f"EXPECTED FINDINGS: {ref['expected_findings']}\n\n"
            f"EVIDENCE SUMMARY:\n{evidence_summary or '(not provided)'}\n\n"
            f"REPORT:\n{report}\n\n"
            "Score the report per the system prompt rubric, in JSON only."
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=0.0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""

        # Parse JSON
        d = {}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                d = {}

        fa = float(d.get("factual_accuracy", 0))
        co = float(d.get("completeness", 0))
        cl = float(d.get("clarity", 0))
        ac = float(d.get("actionability", 0))
        cov = d.get("coverage", []) or []
        cov = [bool(x) for x in cov]
        cov_rate = sum(cov) / len(cov) if cov else 0.0
        overall = (fa + co + cl + ac) / 4.0

        return ReportJudgement(
            question_id=ref["id"],
            factual_accuracy=fa, completeness=co, clarity=cl, actionability=ac,
            overall=overall,
            coverage=cov, coverage_rate=cov_rate,
            overall_comment=d.get("overall_comment", ""),
            expected_empty=expected_empty,
            actually_empty_handled=(empty_handled if expected_empty else True),
        )

    def judge_many(self, reports: list[tuple[dict, str, str]]) -> dict:
        """Score multiple reports.

        Args:
            reports: list of (ref_dict, report_text, evidence_summary) tuples.
        Returns:
            dict with per_question, mean_overall, mean_coverage, empty_pass_rate.
        """
        results = []
        for ref, report, ev in reports:
            results.append(self.judge_one(ref, report, ev))

        n = len(results)
        return {
            "n": n,
            "mean_factual":      sum(r.factual_accuracy for r in results) / n if n else 0.0,
            "mean_completeness": sum(r.completeness     for r in results) / n if n else 0.0,
            "mean_clarity":      sum(r.clarity          for r in results) / n if n else 0.0,
            "mean_actionability":sum(r.actionability    for r in results) / n if n else 0.0,
            "mean_overall":      sum(r.overall          for r in results) / n if n else 0.0,
            "mean_coverage":     sum(r.coverage_rate    for r in results) / n if n else 0.0,
            "empty_pass_rate":
                sum(1 for r in results if r.expected_empty and r.actually_empty_handled)
                / max(1, sum(1 for r in results if r.expected_empty)),
            "per_question": [r.__dict__ for r in results],
        }
