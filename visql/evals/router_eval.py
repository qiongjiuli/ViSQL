"""Router evaluation — 60 hand-labeled + 10 adversarial gate probes.

Slide 10: 0.92 macro-F1, adversarial gate accuracy 8/10. Failures are on
ambiguous "compare X vs Y" phrasings without explicit experiment language.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from visql.router  import TaskRouter
from visql.schemas import DatabaseSchema

# ════════════════════════════════════════════════════════════════════
# DATASET — 60 in-distribution + 10 adversarial
# ════════════════════════════════════════════════════════════════════
# 12 examples × 5 classes = 60 in-distribution items.
ROUTER_EVAL_BASE = [
    # ── single_chart ──
    {"q": "Show me top 10 product categories by revenue.",                        "label": "single_chart"},
    {"q": "Plot daily order count for the past 30 days.",                          "label": "single_chart"},
    {"q": "Bar chart of users by country.",                                        "label": "single_chart"},
    {"q": "Visualize the distribution of order values.",                           "label": "single_chart"},
    {"q": "Top 5 brands by total sales last quarter.",                              "label": "single_chart"},
    {"q": "Show monthly revenue for 2024.",                                        "label": "single_chart"},
    {"q": "Pie chart of traffic sources.",                                          "label": "single_chart"},
    {"q": "Time series of new user signups.",                                      "label": "single_chart"},
    {"q": "Plot the average order value by gender.",                               "label": "single_chart"},
    {"q": "Show top 20 products by units sold.",                                   "label": "single_chart"},
    {"q": "Histogram of customer age.",                                             "label": "single_chart"},
    {"q": "Stacked bar of returns by category.",                                   "label": "single_chart"},
    # ── dashboard ──
    {"q": "Build a dashboard summarizing the e-commerce business.",                "label": "dashboard"},
    {"q": "Give me an overview of all KPIs across users, orders, and inventory.",  "label": "dashboard"},
    {"q": "Create a sales overview dashboard.",                                     "label": "dashboard"},
    {"q": "Marketing dashboard with traffic, conversion, and revenue.",            "label": "dashboard"},
    {"q": "Operations dashboard: inventory, fulfillment, returns.",                "label": "dashboard"},
    {"q": "Executive overview across the whole business.",                         "label": "dashboard"},
    {"q": "Build a dashboard of the top 5 KPIs.",                                  "label": "dashboard"},
    {"q": "Show me a multi-panel summary of last quarter.",                        "label": "dashboard"},
    {"q": "Dashboard of weekly trends across categories.",                         "label": "dashboard"},
    {"q": "Cross-functional dashboard for sales, marketing, ops.",                 "label": "dashboard"},
    {"q": "Quarterly business review dashboard.",                                  "label": "dashboard"},
    {"q": "Dashboard summarizing all order outcomes.",                             "label": "dashboard"},
    # ── ab_test (require schema with variant column) ──
    {"q": "For the checkout_redesign A/B test, did treatment lift conversion?",    "label": "ab_test", "needs_exp_schema": True},
    {"q": "Run an A/B test analysis on the new pricing experiment.",               "label": "ab_test", "needs_exp_schema": True},
    {"q": "Did the variant beat control in the homepage experiment?",              "label": "ab_test", "needs_exp_schema": True},
    {"q": "Compute the lift for the recommendation A/B test.",                     "label": "ab_test", "needs_exp_schema": True},
    {"q": "Was the treatment group's uplift statistically significant?",           "label": "ab_test", "needs_exp_schema": True},
    {"q": "Analyze the randomized experiment on the search ranker.",               "label": "ab_test", "needs_exp_schema": True},
    {"q": "Test conversion rates across treatment vs control.",                    "label": "ab_test", "needs_exp_schema": True},
    {"q": "A/B test results for the new email subject line.",                       "label": "ab_test", "needs_exp_schema": True},
    {"q": "Did treatment X cause higher engagement?",                              "label": "ab_test", "needs_exp_schema": True},
    {"q": "Variant analysis for last month's experiment.",                         "label": "ab_test", "needs_exp_schema": True},
    {"q": "Treatment effect of the redesign experiment.",                          "label": "ab_test", "needs_exp_schema": True},
    {"q": "A/B test: did the new flow lift checkout completion?",                   "label": "ab_test", "needs_exp_schema": True},
    # ── ml_modeling ──
    {"q": "Predict whether an order will be returned.",                            "label": "ml_modeling"},
    {"q": "Build a churn classifier from user behavior.",                          "label": "ml_modeling"},
    {"q": "Forecast next month's revenue.",                                        "label": "ml_modeling"},
    {"q": "Train a model to predict customer lifetime value.",                     "label": "ml_modeling"},
    {"q": "Classify users by likelihood to convert.",                              "label": "ml_modeling"},
    {"q": "Predict order delivery time from order features.",                      "label": "ml_modeling"},
    {"q": "Build a regressor for sale price.",                                     "label": "ml_modeling"},
    {"q": "Train a random forest to predict returns.",                              "label": "ml_modeling"},
    {"q": "Fit a logistic regression to predict purchase.",                        "label": "ml_modeling"},
    {"q": "Predict which users will become repeat buyers.",                        "label": "ml_modeling"},
    {"q": "Model the probability of fraud per order.",                             "label": "ml_modeling"},
    {"q": "Forecast demand for top categories next quarter.",                      "label": "ml_modeling"},
    # ── sql_only ──
    {"q": "Just give me the raw user IDs of users from the UK.",                   "label": "sql_only"},
    {"q": "Extract all order rows from January.",                                  "label": "sql_only"},
    {"q": "List the email addresses of users who signed up last week.",            "label": "sql_only"},
    {"q": "Pull the raw rows for inactive users.",                                  "label": "sql_only"},
    {"q": "Export all transactions over $100.",                                    "label": "sql_only"},
    {"q": "Give me the IDs of products with no inventory.",                         "label": "sql_only"},
    {"q": "Just the list of failed orders.",                                        "label": "sql_only"},
    {"q": "Raw event log for last hour.",                                           "label": "sql_only"},
    {"q": "Dump all rows where status = 'cancelled'.",                              "label": "sql_only"},
    {"q": "Extract user_ids from the experiment table.",                            "label": "sql_only"},
    {"q": "Give me the raw customer addresses.",                                    "label": "sql_only"},
    {"q": "Just the order numbers for refunded orders.",                            "label": "sql_only"},
]

# Adversarial: questions that LOOK like A/B but should be GATED to single_chart
ROUTER_EVAL_ADVERSARIAL = [
    {"q": "Is conversion significantly different between mobile and desktop?",     "label": "single_chart", "adversarial": True},
    {"q": "Compare US vs UK average order value.",                                  "label": "single_chart", "adversarial": True},
    {"q": "Test whether iOS users have higher engagement than Android.",            "label": "single_chart", "adversarial": True},
    {"q": "Are returning customers different from new customers in spend?",         "label": "single_chart", "adversarial": True},
    {"q": "Did men spend more than women last quarter?",                            "label": "single_chart", "adversarial": True},
    {"q": "Is there a significant difference between weekday and weekend orders?",  "label": "single_chart", "adversarial": True},
    {"q": "Compare conversion: paid vs organic search.",                            "label": "single_chart", "adversarial": True},
    {"q": "Treatment effect of being in California vs New York on order size.",     "label": "single_chart", "adversarial": True},
    {"q": "Did segment A convert at a higher rate than segment B last month?",      "label": "single_chart", "adversarial": True},
    {"q": "Compare repeat customers and one-time customers.",                       "label": "single_chart", "adversarial": True},
]


# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
@dataclass
class RouterEvalSummary:
    n: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class_precision: dict = field(default_factory=dict)
    per_class_recall: dict = field(default_factory=dict)
    per_class_f1: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)
    ab_gate_correct: int = 0
    ab_gate_total: int = 0
    adversarial_correct: int = 0
    adversarial_total: int = 0
    mistakes: list = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# EVALUATOR
# ════════════════════════════════════════════════════════════════════
class RouterEvaluator:
    def __init__(self,
                 router: TaskRouter,
                 schema_with_exp: DatabaseSchema,
                 schema_without_exp: DatabaseSchema,
                 base_set: Optional[list] = None,
                 adversarial_set: Optional[list] = None):
        self.router = router
        self.schema_with_exp = schema_with_exp
        self.schema_without_exp = schema_without_exp
        self.base_set = base_set or ROUTER_EVAL_BASE
        self.adversarial_set = adversarial_set or ROUTER_EVAL_ADVERSARIAL

    def evaluate(self, verbose: bool = False) -> RouterEvalSummary:
        items = list(self.base_set) + list(self.adversarial_set)
        confusion = defaultdict(lambda: defaultdict(int))
        mistakes = []
        ab_gate_correct = ab_gate_total = 0
        adv_correct = adv_total = 0

        for ex in items:
            q = ex["q"]
            true = ex["label"]
            schema = (self.schema_with_exp if ex.get("needs_exp_schema")
                      else self.schema_without_exp)
            decision = self.router.route(q, schema)
            pred = decision.label
            confusion[true][pred] += 1
            if pred != true:
                mistakes.append({"q": q, "true": true, "pred": pred,
                                 "rationale": decision.rationale,
                                 "gated_from": decision.gated_from})

            if true == "ab_test":
                ab_gate_total += 1
                if pred == "ab_test":
                    ab_gate_correct += 1
            if ex.get("adversarial"):
                adv_total += 1
                if pred == true:
                    adv_correct += 1

            if verbose:
                ok = "✓" if pred == true else "✗"
                print(f"{ok} [{true:>14s} / {pred:>14s}]  {q}")

        # macro-F1
        labels = set(confusion.keys()) | {p for d in confusion.values() for p in d}
        prec, rec, f1 = {}, {}, {}
        for c in labels:
            tp = confusion[c][c]
            fp = sum(confusion[k][c] for k in confusion if k != c)
            fn = sum(confusion[c][k] for k in confusion[c] if k != c)
            prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec[c]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1[c]   = (2 * prec[c] * rec[c] / (prec[c] + rec[c])
                       if (prec[c] + rec[c]) > 0 else 0.0)

        n_correct = sum(confusion[c][c] for c in confusion)
        n = sum(confusion[t][p] for t in confusion for p in confusion[t])

        return RouterEvalSummary(
            n=n,
            accuracy=n_correct / n if n else 0.0,
            macro_f1=sum(f1.values()) / len(f1) if f1 else 0.0,
            per_class_precision=dict(prec),
            per_class_recall=dict(rec),
            per_class_f1=dict(f1),
            confusion={t: dict(d) for t, d in confusion.items()},
            ab_gate_correct=ab_gate_correct,
            ab_gate_total=ab_gate_total,
            adversarial_correct=adv_correct,
            adversarial_total=adv_total,
            mistakes=mistakes,
        )
