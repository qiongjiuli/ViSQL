"""Evaluation suites for ViSQL v2 (slide 10).

Four suites, each tests a different failure mode:
    - router_eval  : task classification + adversarial A/B gate
    - style_eval   : multimodal style imitation, ΔE-76 self-consistency
    - report_eval  : LLM-as-judge with 4-axis rubric
    - sql_eval     : Spider 1.0 dev execution accuracy
"""
from .router_eval import RouterEvaluator, RouterEvalSummary
from .style_eval  import StyleEvaluator, StyleEvalSummary
from .report_eval import ReportJudge, ReportJudgement, REFERENCE_QUESTIONS
from .sql_eval    import SpiderEvaluator, SpiderEvalSummary

__all__ = [
    "RouterEvaluator", "RouterEvalSummary",
    "StyleEvaluator", "StyleEvalSummary",
    "ReportJudge", "ReportJudgement", "REFERENCE_QUESTIONS",
    "SpiderEvaluator", "SpiderEvalSummary",
]
