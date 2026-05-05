"""SQL eval — Spider 1.0 dev execution accuracy.

Slide 10: 0.74 execution accuracy on 200-example slice. The LoRA-tuned
SQLCoder-7B-2 from slide 7 is the model evaluated; the +13pp delta is over
the un-tuned base (0.61).

Execution accuracy: a generated SQL is correct if executing both it and the
gold SQL on the same SQLite DB yields the same set of rows.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import json
import sqlite3


@dataclass
class SpiderEvalSummary:
    n: int = 0
    n_correct: int = 0
    exec_accuracy: float = 0.0
    syntax_errors: int = 0
    runtime_errors: int = 0
    per_example: list = field(default_factory=list)


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


class SpiderEvaluator:
    """Spider 1.0 dev execution-accuracy evaluator."""

    def __init__(self,
                 spider_dev_path: str | Path,
                 spider_db_root: str | Path,
                 generate_fn: Callable[[str, str], str],
                 max_examples: Optional[int] = 200):
        """
        Args:
            spider_dev_path: spider/dev.json
            spider_db_root : spider/database/ (contains <db_id>/<db_id>.sqlite)
            generate_fn    : callable (question, db_id) -> SQL string
            max_examples   : cap evaluation set size (slide 10 used 200)
        """
        self.dev_path = Path(spider_dev_path)
        self.db_root = Path(spider_db_root)
        self.generate_fn = generate_fn
        self.max_examples = max_examples

    def evaluate(self, verbose: bool = False) -> SpiderEvalSummary:
        with open(self.dev_path) as f:
            data = json.load(f)
        if self.max_examples:
            data = data[:self.max_examples]

        per = []
        n_correct = syntax_err = runtime_err = 0
        for i, ex in enumerate(data):
            db_id = ex["db_id"]
            db_path = self.db_root / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                continue
            question = ex["question"]
            gold_sql = ex["query"]
            try:
                pred_sql = self.generate_fn(question, db_id)
            except Exception as e:
                syntax_err += 1
                per.append({"i": i, "db_id": db_id, "q": question,
                            "gold": gold_sql, "pred": "",
                            "match": False, "error": f"gen: {e}"})
                continue

            gold_rows, gold_err = _exec(db_path, gold_sql)
            if gold_err is not None:
                # gold itself is broken — skip
                per.append({"i": i, "db_id": db_id, "q": question,
                            "gold": gold_sql, "pred": pred_sql,
                            "match": None, "error": f"gold: {gold_err}"})
                continue

            pred_rows, pred_err = _exec(db_path, pred_sql)
            if pred_err is not None:
                runtime_err += 1
                per.append({"i": i, "db_id": db_id, "q": question,
                            "gold": gold_sql, "pred": pred_sql,
                            "match": False, "error": f"runtime: {pred_err}"})
                continue

            match = (_norm_rows(pred_rows) == _norm_rows(gold_rows))
            if match:
                n_correct += 1
            per.append({"i": i, "db_id": db_id, "q": question,
                        "gold": gold_sql, "pred": pred_sql,
                        "match": bool(match), "error": None})
            if verbose:
                print(f"  [{i:>3d}] {'✓' if match else '✗'}  {question[:80]}")

        n = len([p for p in per if p["match"] is not None])
        return SpiderEvalSummary(
            n=n,
            n_correct=n_correct,
            exec_accuracy=n_correct / n if n else 0.0,
            syntax_errors=syntax_err,
            runtime_errors=runtime_err,
            per_example=per,
        )
