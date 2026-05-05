"""Stage 5 — Analysis branches + Stage 6 — Report writer.

Five branches (matching slide 5: "5 branches dispatched"):
  - ab_test       : chi² + lift confidence interval
  - fit_linear    : logistic / linear regression with sklearn
  - fit_tree      : random forest with feature importances
  - fit_nn        : small MLP with gradient-attribution top features
  - (single_chart and dashboard branches live in renderer.py)
  - (sql_only just returns the dataframe)

ReportWriter implements the EMPTY-DATA GUARD from slide 4: if the SQL returns
zero rows, the writer refuses to narrate and returns the executed SQL +
diagnostic instead of fabricating findings from training data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import textwrap

import numpy as np
import pandas as pd

from . import config as cfg

# ════════════════════════════════════════════════════════════════════
# A/B TEST BRANCH
# ════════════════════════════════════════════════════════════════════
@dataclass
class ABTestResult:
    n_control: int = 0
    n_treatment: int = 0
    conv_control: float = 0.0
    conv_treatment: float = 0.0
    lift_pp: float = 0.0
    lift_ci_low: float = 0.0
    lift_ci_high: float = 0.0
    chi2: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    summary: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}


def ab_test(df: pd.DataFrame,
            variant_col: str = "variant",
            outcome_col: str = "converted",
            control_label: str = "control",
            treatment_label: str = "treatment",
            alpha: float = 0.05) -> ABTestResult:
    """chi² test on a 2x2 contingency table + lift CI.

    Used for the A/B test demo on slide 9 (chi² = 412.7, p < .001, lift = +8.1pp).
    """
    from scipy import stats

    if variant_col not in df.columns or outcome_col not in df.columns:
        return ABTestResult(summary=f"Required columns missing: {variant_col}, {outcome_col}")

    sub = df[df[variant_col].isin([control_label, treatment_label])].copy()
    n_c = (sub[variant_col] == control_label).sum()
    n_t = (sub[variant_col] == treatment_label).sum()

    if n_c == 0 or n_t == 0:
        return ABTestResult(summary="Empty control or treatment group.")

    conv_c = sub.loc[sub[variant_col] == control_label, outcome_col].mean()
    conv_t = sub.loc[sub[variant_col] == treatment_label, outcome_col].mean()
    lift_pp = (conv_t - conv_c) * 100.0

    # chi² on counts
    succ_c = sub.loc[sub[variant_col] == control_label, outcome_col].sum()
    succ_t = sub.loc[sub[variant_col] == treatment_label, outcome_col].sum()
    contingency = np.array([[succ_c, n_c - succ_c],
                            [succ_t, n_t - succ_t]])
    chi2, p, _, _ = stats.chi2_contingency(contingency, correction=False)

    # Wald CI on the lift (in percentage points)
    se = math.sqrt(conv_c * (1 - conv_c) / n_c + conv_t * (1 - conv_t) / n_t)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low_pp = lift_pp - z * se * 100.0
    ci_high_pp = lift_pp + z * se * 100.0

    significant = p < alpha and lift_pp != 0
    summary = (
        f"chi² = {chi2:.1f}, p = {p:.4g}, lift = {lift_pp:+.2f}pp "
        f"(95% CI: [{ci_low_pp:+.2f}, {ci_high_pp:+.2f}]). "
        f"Control: {conv_c:.3f} (n={n_c:,}). Treatment: {conv_t:.3f} (n={n_t:,})."
    )
    return ABTestResult(
        n_control=int(n_c), n_treatment=int(n_t),
        conv_control=float(conv_c), conv_treatment=float(conv_t),
        lift_pp=float(lift_pp),
        lift_ci_low=float(ci_low_pp), lift_ci_high=float(ci_high_pp),
        chi2=float(chi2), p_value=float(p),
        significant=bool(significant),
        summary=summary,
    )

# ════════════════════════════════════════════════════════════════════
# ML MODELING BRANCHES (linear / tree / NN)
# ════════════════════════════════════════════════════════════════════
@dataclass
class ModelResult:
    model_type: str = ""
    target: str = ""
    features: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    top_features: dict = field(default_factory=dict)
    summary: str = ""


def _split_xy(df: pd.DataFrame, target: str, features: list[str]):
    """Encode features, return numpy X, y, encoded feature names."""
    df = df[features + [target]].dropna().copy()
    y = df[target].values
    X = pd.get_dummies(df[features], drop_first=False, dummy_na=False)
    feat_names = X.columns.tolist()
    return X.values.astype(float), y, feat_names


def fit_linear(df: pd.DataFrame, target: str, features: list[str]) -> ModelResult:
    """Logistic/linear regression depending on target type."""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

    X, y, feat_names = _split_xy(df, target, features)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    is_classification = (len(np.unique(y)) == 2)
    if is_classification:
        m = LogisticRegression(max_iter=1000)
        m.fit(X_std, y)
        preds = m.predict(X_std)
        try:
            auc = float(roc_auc_score(y, m.predict_proba(X_std)[:, 1]))
        except Exception:
            auc = float("nan")
        metrics = {"accuracy": float(accuracy_score(y, preds)), "auc": auc}
        coef = m.coef_[0]
    else:
        m = LinearRegression()
        m.fit(X_std, y)
        metrics = {"r2": float(r2_score(y, m.predict(X_std)))}
        coef = m.coef_

    # Top features by absolute standardized coefficient
    order = np.argsort(-np.abs(coef))[:5]
    top_feats = {feat_names[i]: float(coef[i]) for i in order}
    summary = (
        f"Linear model ({'logistic' if is_classification else 'OLS'}) on {target}. "
        f"Metrics: {metrics}. Top effects (standardized): "
        + ", ".join(f"{k}={v:+.3f}" for k, v in top_feats.items())
    )
    return ModelResult(
        model_type="linear", target=target, features=features,
        metrics=metrics, top_features=top_feats, summary=summary,
    )


def fit_tree(df: pd.DataFrame, target: str, features: list[str]) -> ModelResult:
    """Random forest with feature importances."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

    X, y, feat_names = _split_xy(df, target, features)
    is_classification = (len(np.unique(y)) == 2)
    if is_classification:
        m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        m.fit(X, y)
        try:
            auc = float(roc_auc_score(y, m.predict_proba(X)[:, 1]))
        except Exception:
            auc = float("nan")
        metrics = {"accuracy": float(accuracy_score(y, m.predict(X))), "auc": auc}
    else:
        m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        m.fit(X, y)
        metrics = {"r2": float(r2_score(y, m.predict(X)))}

    imp = m.feature_importances_
    order = np.argsort(-imp)[:5]
    top_feats = {feat_names[i]: float(imp[i]) for i in order}
    summary = (
        f"Random forest ({'classifier' if is_classification else 'regressor'}) on {target}. "
        f"Metrics: {metrics}. Top importances: "
        + ", ".join(f"{k}={v:.3f}" for k, v in top_feats.items())
    )
    return ModelResult(
        model_type="tree", target=target, features=features,
        metrics=metrics, top_features=top_feats, summary=summary,
    )


def fit_nn(df: pd.DataFrame, target: str, features: list[str],
           epochs: int = 30) -> ModelResult:
    """Small MLP with gradient-attribution feature importance."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

    X, y, feat_names = _split_xy(df, target, features)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    Xt = torch.tensor(X_std, dtype=torch.float32)
    is_classification = (len(np.unique(y)) == 2)
    yt = torch.tensor(y, dtype=torch.float32 if not is_classification else torch.long)

    in_dim = X_std.shape[1]
    out_dim = 2 if is_classification else 1

    class MLP(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, d_out),
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(in_dim, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        if is_classification:
            loss = loss_fn(pred, yt)
        else:
            loss = loss_fn(pred.squeeze(), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(Xt)
        if is_classification:
            yhat = out.argmax(dim=1).numpy()
            try:
                probs = torch.softmax(out, dim=1)[:, 1].numpy()
                auc = float(roc_auc_score(y, probs))
            except Exception:
                auc = float("nan")
            metrics = {"accuracy": float(accuracy_score(y, yhat)), "auc": auc}
        else:
            metrics = {"r2": float(r2_score(y, out.squeeze().numpy()))}

    # gradient-attribution feature importance (mean |∂loss/∂x|)
    Xt.requires_grad_(True)
    out = model(Xt)
    target_score = out.sum() if not is_classification else out[:, 1].sum()
    target_score.backward()
    grads = Xt.grad.detach().abs().mean(dim=0).numpy()
    order = np.argsort(-grads)[:5]
    top_feats = {feat_names[i]: float(grads[i]) for i in order}
    summary = (
        f"MLP ({'classifier' if is_classification else 'regressor'}) on {target}, "
        f"trained {epochs} epochs. Metrics: {metrics}. "
        f"Top |grad|-attributions: " + ", ".join(f"{k}={v:.3f}" for k, v in top_feats.items())
    )
    return ModelResult(
        model_type="nn", target=target, features=features,
        metrics=metrics, top_features=top_feats, summary=summary,
    )

# ════════════════════════════════════════════════════════════════════
# REPORT WRITER (Stage 6) — with EMPTY-DATA GUARD (slide 4)
# ════════════════════════════════════════════════════════════════════
REPORT_SYSTEM = textwrap.dedent("""
    You are a senior data analyst writing a brief analytical report.

    Rules:
    - Cite SPECIFIC numbers from the evidence given. Do not invent statistics.
    - 2-4 short paragraphs, ~250 words total. No markdown headings.
    - Open with the answer, then evidence, then caveats.
    - Plain text only.

    OUTPUT FORMAT (CRITICAL — the rendering layer treats $ as math delimiter):
    - Always escape currency dollar signs as \\$ when writing dollar amounts
      (e.g. write \\$1.47 million, NOT $1.47 million).
""").strip()


class ReportWriter:
    """Stage 6 — Claude-backed narrative generator with empty-data guard."""

    def __init__(self, anthropic_client, model: str = cfg.CLAUDE_MODEL):
        self.client = anthropic_client
        self.model = model

    def write(self,
              question: str,
              df: Optional[pd.DataFrame],
              executed_sql: str,
              plan: Optional[dict] = None,
              ab_result: Optional[ABTestResult] = None,
              model_result: Optional[ModelResult] = None,
              extra_context: str = "") -> str:
        """Generate a grounded report.

        EMPTY-DATA GUARD: if df is None or empty, refuse to narrate and return
        a diagnostic instead. This is the failure mode flagged on slide 4 —
        early iterations would fabricate findings from training data when the
        SQL returned no rows.
        """
        # Guard 1 — no df at all
        if df is None:
            return (
                "No data was returned because the SQL did not execute successfully.\n\n"
                f"Question: {question}\n\nAttempted SQL:\n{executed_sql}\n\n"
                "I cannot narrate findings without an executed result; please review "
                "the SQL or schema and retry."
            )

        # Guard 2 — df is empty
        if len(df) == 0:
            return (
                "The query executed but returned 0 rows. I'm declining to write a "
                "narrative report from this — generating a story from no data risks "
                "fabricating findings.\n\n"
                f"Question: {question}\n\nExecuted SQL:\n{executed_sql}\n\n"
                "Likely causes: filter too restrictive, date range out of bounds, "
                "or join condition mismatched. Please refine the question or check "
                "the schema and retry."
            )

        # Build evidence block
        ev_parts = [f"QUESTION: {question}", "", f"EXECUTED SQL:\n{executed_sql}", ""]
        if plan:
            ev_parts.append(f"PLAN: {plan.get('intent', '(none)')}")
            ev_parts.append("")
        if ab_result is not None:
            ev_parts.append(f"A/B TEST RESULT: {ab_result.summary}")
            ev_parts.append("")
        if model_result is not None:
            ev_parts.append(f"MODEL RESULT: {model_result.summary}")
            ev_parts.append("")
        # Up to 30 rows of data, truncated for prompt size
        ev_parts.append("DATA (first rows):")
        ev_parts.append(df.head(30).to_string(index=False, max_cols=10))
        if extra_context:
            ev_parts.append(f"\nEXTRA CONTEXT:\n{extra_context}")
        evidence = "\n".join(ev_parts)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=900,
            temperature=0.0,
            system=REPORT_SYSTEM,
            messages=[{"role": "user", "content": evidence}],
        )
        text = resp.content[0].text if resp.content else "(empty response)"
        return text.strip()
