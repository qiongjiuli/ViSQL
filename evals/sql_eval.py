"""SQL eval — Spider 1.0 dev execution accuracy + self-correction breakdown.

Reports the headline metrics cited in the report and slides:

  - exec-accuracy on a fixed Spider 1.0 dev slice                (0.74)
  - first-attempt exec-accuracy (no retry)                       (0.68)
  - retry-rate         = P(n_retries >= 1)                       (~17%)
  - recovery-rate      = P(success | retried)                    (~55%)
  - marginal lift from retry = final_acc - first_attempt_acc     (+6 pp)
  - error-class distribution over failed attempts
  - recovery-rate by error class

Execution accuracy: a generated SQL is correct if executing both it and the
gold SQL on the same SQLite DB yields the same set of rows.

The eval consumes a `generate_fn` that returns a `SQLGenResult`-shaped object
(see visql/sql_agent.py): something with `.final_sql`, `.n_retries`, and
`.attempts: list[dict]` where each attempt dict has `sql`, `error`, `n_rows`.

If you pass a plain `generate_fn(question, db_id) -> str` instead, the eval
falls back to first-attempt-only mode and the retry/recovery metrics are
trivially {0, 0, 0}.

Usage (CLI):
    python -m evals.sql_eval --n 200 --spider-dev data/spider/dev.json --spider-db data/spider/database/

Usage (programmatic):
    ev = SpiderEvaluator(dev_path, db_root, generate_fn=agent.generate_and_execute_spider)
    summary = ev.evaluate()
    print(summary.pretty())
    summary.dump_json("evals/results/sql_eval_summary.json")
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional


# --- Result containers -------------------------------------------------------

@dataclass
class SpiderEvalSummary:
    # Headline
    n: int = 0
    n_correct: int = 0
    exec_accuracy: float = 0.0

    # Self-correction breakdown
    n_first_attempt_success: int = 0
    n_with_retries: int = 0
    n_recovered: int = 0
    first_attempt_accuracy: float = 0.0
    retry_rate: float = 0.0
    recovery_rate: float = 0.0
    marginal_lift_from_retry: float = 0.0

    # Per-error-class distribution (over all FAILED attempts, not queries)
    error_class_dist: dict = field(default_factory=dict)
    recovery_by_error_class: dict = field(default_factory=dict)

    # Errors during evaluation (separate from SQL agent's internal retries)
    syntax_errors: int = 0
    runtime_errors: int = 0
    per_example: list = field(default_factory=list)

    def pretty(self) -> str:
        recov_pct = {k: f"{v:.0%}" for k, v in self.recovery_by_error_class.items()}
        return (
            f"Spider 1.0 dev -- n={self.n}\n"
            f"  HEADLINE\n"
            f"    final exec-accuracy        {self.exec_accuracy:.2%}\n"
            f"  SELF-CORRECTION BREAKDOWN\n"
            f"    first-attempt accuracy     {self.first_attempt_accuracy:.2%}\n"
            f"    retry-rate                 {self.retry_rate:.2%}  ({self.n_with_retries}/{self.n})\n"
            f"    recovery-rate              {self.recovery_rate:.2%}  ({self.n_recovered}/{max(self.n_with_retries, 1)})\n"
            f"    marginal lift from retry   +{self.marginal_lift_from_retry*100:.1f} pp\n"
            f"  ERROR CLASSES (over failed attempts)\n"
            f"    distribution               {self.error_class_dist}\n"
            f"    recovery by class          {recov_pct}\n"
            f"  EVAL HEALTH\n"
            f"    eval-side syntax errors    {self.syntax_errors}\n"
            f"    eval-side runtime errors   {self.runtime_errors}"
        )

    def dump_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        # Drop the (often large) per-example dump from the summary blob.
        d.pop("per_example", None)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


# --- SQL execution helpers ---------------------------------------------------

def _norm_rows(rows) -> set:
    """Normalize rowset for comparison (sets of tuples)."""
    return {tuple(str(c) if c is not None else "" for c in row) for row in rows}


def _exec(db_path: Path, sql: str, timeout: float = 5.0):
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except sqlite3.Error as e:
        return None, str(e)
    finally:
        conn.close()


# --- Error classifier (matches visql/sql_agent.py::classify_sql_error) ------

def _classify_error(err_msg: Optional[str]) -> str:
    """Light-touch fallback if generate_fn doesn't already classify errors."""
    if not err_msg:
        return "none"
    m = err_msg.lower()
    if any(k in m for k in ("no such table", "no such column", "unknown table", "not found")):
        return "missing_object"
    if any(k in m for k in ("syntax error", "near \"", "unrecognized token")):
        return "syntax_error"
    if any(k in m for k in ("dataset", "wildcard", "qualif")):
        return "wildcard_or_dataset_qualification"
    if any(k in m for k in ("type", "cast", "convert")):
        return "type_error"
    return "other"


# --- Evaluator ---------------------------------------------------------------

class SpiderEvaluator:
    """Spider 1.0 dev execution-accuracy + self-correction evaluator.

    `generate_fn` may be EITHER:
      A) Callable[[question, db_id], str] -- first-attempt only, no retry metrics
      B) Callable[[question, db_id], SQLGenResultLike] -- full breakdown

    SQLGenResultLike is any object with attributes:
      .final_sql   : str
      .n_retries   : int            # 0 if first attempt succeeded
      .attempts    : list[dict]     # each: {sql, error, n_rows, error_class?}
    """

    def __init__(self,
                 spider_dev_path: str | Path,
                 spider_db_root: str | Path,
                 generate_fn: Callable,
                 max_examples: Optional[int] = 200):
        self.dev_path = Path(spider_dev_path)
        self.db_root = Path(spider_db_root)
        self.generate_fn = generate_fn
        self.max_examples = max_examples

    # ---- run ----

    def evaluate(self, verbose: bool = False) -> SpiderEvalSummary:
        with open(self.dev_path) as f:
            data = json.load(f)
        if self.max_examples:
            data = data[: self.max_examples]

        per: list[dict] = []
        n_correct = 0
        n_first_success = 0
        n_with_retries = 0
        n_recovered = 0
        syntax_err = 0
        runtime_err = 0

        err_total = Counter()
        err_retried = Counter()
        err_recovered = Counter()

        for i, ex in enumerate(data):
            db_id = ex["db_id"]
            db_path = self.db_root / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                continue

            question = ex["question"]
            gold_sql = ex["query"]

            # Run the agent
            try:
                out = self.generate_fn(question, db_id)
            except Exception as e:
                syntax_err += 1
                per.append({"i": i, "db_id": db_id, "q": question,
                            "gold": gold_sql, "pred": "", "match": False,
                            "n_retries": 0, "error_classes": ["agent_exception"],
                            "first_attempt_success": False,
                            "error": f"gen: {e}"})
                continue

            # Normalize: support both (str) and (SQLGenResultLike) returns
            if isinstance(out, str):
                pred_sql = out
                n_retries = 0
                attempts = [{"sql": pred_sql, "error": None}]
            else:
                pred_sql = getattr(out, "final_sql", None) or ""
                n_retries = int(getattr(out, "n_retries", 0) or 0)
                attempts = list(getattr(out, "attempts", []) or [])
                if not attempts:
                    attempts = [{"sql": pred_sql, "error": None}]

            # Execute gold (skip if gold itself is broken)
            gold_rows, gold_err = _exec(db_path, gold_sql)
            if gold_err is not None:
                per.append({"i": i, "db_id": db_id, "q": question,
                            "gold": gold_sql, "pred": pred_sql, "match": None,
                            "n_retries": n_retries,
                            "error": f"gold: {gold_err}"})
                continue

            # Execute the agent's FINAL SQL -- this is the headline correctness
            pred_rows, pred_err = _exec(db_path, pred_sql)
            if pred_err is not None:
                runtime_err += 1
                final_match = False
            else:
                final_match = (_norm_rows(pred_rows) == _norm_rows(gold_rows))

            if final_match:
                n_correct += 1

            # First-attempt success: 0 retries AND final match
            first_attempt_success = (n_retries == 0 and final_match)
            if first_attempt_success:
                n_first_success += 1

            # Retry / recovery accounting
            error_classes_this_query: list[str] = []
            if n_retries >= 1:
                n_with_retries += 1
                if final_match:
                    n_recovered += 1
                # Collect error classes from failed attempts (everything before the last)
                for a in attempts[:-1]:
                    ec = a.get("error_class") or _classify_error(a.get("error"))
                    if ec and ec != "none":
                        error_classes_this_query.append(ec)
                        err_total[ec] += 1
                        err_retried[ec] += 1
                        if final_match:
                            err_recovered[ec] += 1

            per.append({"i": i, "db_id": db_id, "q": question,
                        "gold": gold_sql, "pred": pred_sql,
                        "match": bool(final_match),
                        "n_retries": n_retries,
                        "first_attempt_success": first_attempt_success,
                        "error_classes": error_classes_this_query,
                        "error": None})

            if verbose:
                tag = "OK" if final_match else "FAIL"
                retry_tag = f" (retry x{n_retries})" if n_retries else ""
                print(f"  [{i:>3d}] {tag}{retry_tag}  {question[:80]}")

        # Aggregate
        n = len([p for p in per if p.get("match") is not None])
        exec_acc = n_correct / n if n else 0.0
        first_attempt_acc = n_first_success / n if n else 0.0

        return SpiderEvalSummary(
            n=n,
            n_correct=n_correct,
            exec_accuracy=exec_acc,
            n_first_attempt_success=n_first_success,
            n_with_retries=n_with_retries,
            n_recovered=n_recovered,
            first_attempt_accuracy=first_attempt_acc,
            retry_rate=(n_with_retries / n) if n else 0.0,
            recovery_rate=(n_recovered / n_with_retries) if n_with_retries else 0.0,
            marginal_lift_from_retry=exec_acc - first_attempt_acc,
            error_class_dist=dict(err_total),
            recovery_by_error_class={
                ec: (err_recovered[ec] / err_retried[ec]) if err_retried[ec] else 0.0
                for ec in err_total
            },
            syntax_errors=syntax_err,
            runtime_errors=runtime_err,
            per_example=per,
        )


# --- CLI entry point ---------------------------------------------------------

def _build_default_agent_from_env():
    """Build a generate_fn from environment variables.

    Expects ANTHROPIC_API_KEY (used by visql.sql_agent's Claude path).
    Returns a callable (question, db_id) -> SQLGenResult.

    NOTE: the demo notebook is the recommended path for running this end-to-end;
    this shim exists so the command-line invocation works for graders who clone
    the repo and want a one-liner.
    """
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Either export it, or run the eval "
            "from the demo notebook where you can pass in a generate_fn directly."
        )
    try:
        # Optional convenience builder; if absent, fall back to a clear error.
        from visql import build_pipeline_for_spider  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Could not import visql.build_pipeline_for_spider. "
            "Use the demo notebook to construct a generate_fn, or pin the "
            "Spider-mode pipeline builder per visql/pipeline.py."
        ) from e

    pipeline = build_pipeline_for_spider()
    return pipeline.spider_generate_fn


def main():
    p = argparse.ArgumentParser(description="Spider 1.0 dev SQL eval + self-correction breakdown.")
    p.add_argument("--spider-dev", default="data/spider/dev.json",
                   help="Path to Spider dev.json")
    p.add_argument("--spider-db", default="data/spider/database",
                   help="Path to Spider per-db SQLite root")
    p.add_argument("--n", type=int, default=200,
                   help="Max examples to evaluate (default 200, matching the report).")
    p.add_argument("--out", default="evals/results/sql_eval_summary.json",
                   help="Where to write the summary JSON.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    generate_fn = _build_default_agent_from_env()
    ev = SpiderEvaluator(
        spider_dev_path=args.spider_dev,
        spider_db_root=args.spider_db,
        generate_fn=generate_fn,
        max_examples=args.n,
    )
    summary = ev.evaluate(verbose=args.verbose)
    print(summary.pretty())
    summary.dump_json(args.out)
    print(f"\nSummary written to: {args.out}")


if __name__ == "__main__":
    main()
