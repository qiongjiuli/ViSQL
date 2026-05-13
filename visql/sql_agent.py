"""Stage 4 — SQL agent.

Backed by Claude Sonnet 4.5. Implements the architectural elements from
slide 3 (stage 4) and slide 7 (few-shot retrieval):

  - Few-shot exemplars from Spider 1.0 (k=3) injected into the prompt.
  - BigQuery dialect compliance enforced explicitly in the system prompt.
  - PRE-EXECUTION table-existence check (slide 4) — catches hallucinated
    tables before paying for the BQ round-trip.
  - SELF-CORRECTION loop: up to 3 retries on execution error, with the
    error message fed back into the next attempt.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re

from . import config as cfg
from .schemas import DatabaseSchema, DatabaseManager, check_tables_exist
from .planner import QueryPlan
from .retrievers import ExemplarRetriever

@dataclass
class SQLGenResult:
    sql: str = ""
    final_sql: str = ""
    df: Optional[object] = None        # pandas DataFrame after execution
    n_rows: int = 0
    n_retries: int = 0
    error: Optional[str] = None
    attempts: list[dict] = field(default_factory=list)
    exemplars_used: list[dict] = field(default_factory=list)

# ── Error classifier (used to give better retry hints) ────────────
def classify_sql_error(err: str) -> str:
    """Heuristic classifier so we can give the model a useful retry hint."""
    e = err.lower()
    if "not found" in e or "does not exist" in e or "unrecognized name" in e:
        return "missing_object"
    if "syntax error" in e or "unexpected" in e:
        return "syntax_error"
    if "cannot be qualified" in e or "must be qualified" in e:
        return "wildcard_or_dataset_qualification"
    if "type" in e and ("mismatch" in e or "cannot" in e):
        return "type_error"
    if "permission" in e or "access denied" in e:
        return "permission"
    return "other"

# ── System prompt ─────────────────────────────────────────────────
SQL_SYSTEM = """You are a SQL agent for Google BigQuery. You write a single, executable BigQuery Standard SQL query for the user's question.

ABSOLUTE RULES:
1. BigQuery Standard SQL ONLY. Never use PostgreSQL syntax: ::CASTS, RETURNING, ILIKE, LIMIT n OFFSET m without ORDER BY, etc. Use SAFE_CAST(x AS TYPE), CAST(x AS TYPE), or PARSE_DATE.
2. Always fully-qualify tables with backticks: `project.dataset.table`. Use the fq_name shown in the schema.
3. Always include a LIMIT clause (≤ 1000) unless the user explicitly asks for all rows.
4. Use SAFE_DIVIDE for division to avoid zero-division errors.
5. For dates, use PARSE_DATE / PARSE_TIMESTAMP / DATE_TRUNC, not PostgreSQL's ::DATE.
6. Output the SQL ONLY, inside a single ```sql code block. No prose, no commentary.
"""

# ════════════════════════════════════════════════════════════════════
# SQL AGENT
# ════════════════════════════════════════════════════════════════════
class SQLAgent:
    """Stage 4 — Claude-backed SQL generator with few-shot retrieval, table check, and retry loop."""

    def __init__(self,
                 anthropic_client,
                 exemplar_retriever: Optional[ExemplarRetriever] = None,
                 model: str = cfg.CLAUDE_MODEL,
                 max_retries: int = cfg.SQL_MAX_RETRIES):
        self.client = anthropic_client
        self.exemplars = exemplar_retriever
        self.model = model
        self.max_retries = max_retries

    # ── Prompt construction ──────────────────────────────────────
    def _build_user_prompt(self, question: str, plan: QueryPlan,
                            schema: DatabaseSchema, retry_hint: str = "") -> tuple[str, list[dict]]:
        schema_block = schema.to_prompt_block(max_chars=4500)

        # Few-shot block — only when we have exemplars
        few_shot_block = ""
        used_exemplars = []
        if self.exemplars is not None:
            try:
                used_exemplars = self.exemplars.retrieve(question, k=cfg.TOP_K_EXEMPLARS)
                if used_exemplars:
                    few_shot_block = "\nFEW-SHOT EXAMPLES (Spider patterns; ignore the schemas, use the patterns):\n"
                    for i, ex in enumerate(used_exemplars, 1):
                        few_shot_block += f"\nEx{i}: {ex['question']}\nSQL: {ex['sql']}\n"
            except Exception:
                pass

        # Plan block
        plan_block = (
            "\nPLAN (from CoT planner):\n"
            f"  intent: {plan.intent}\n"
            f"  required_tables: {plan.required_tables}\n"
            f"  aggregations: {plan.aggregations}\n"
            f"  grouping: {plan.grouping}\n"
            f"  ordering: {plan.ordering}\n"
        )

        retry_block = ""
        if retry_hint:
            retry_block = f"\nRETRY CONTEXT — your previous attempt failed:\n{retry_hint}\nWrite a corrected query.\n"

        user = (
            f"SCHEMA:\n{schema_block}\n"
            f"{plan_block}"
            f"{few_shot_block}"
            f"{retry_block}\n"
            f"QUESTION: {question}\n\nWrite the SQL now."
        )
        return user, used_exemplars

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Pull the SQL out of a ```sql ...``` block (or any plausible block)."""
        m = re.search(r"```(?:sql|SQL)?\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip().rstrip(";") + ";"
        # fallback: first SELECT/WITH onwards
        for kw in ("WITH", "SELECT"):
            i = text.upper().find(kw)
            if i >= 0:
                tail = text[i:].strip()
                return (tail.split(";")[0].strip() + ";") if ";" in tail else tail
        return text.strip()

    # ── Single Claude call ───────────────────────────────────────
    def _call_claude(self, user_prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0.0,
            system=SQL_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text if resp.content else ""

    # ── Main entrypoint ─────────────────────────────────────────
    def generate_and_execute(
        self,
        question: str,
        plan: QueryPlan,
        schema: DatabaseSchema,
        db_manager: DatabaseManager,
    ) -> SQLGenResult:
        """Generate SQL, validate it, execute it, retry on error up to N times."""
        result = SQLGenResult()
        retry_hint = ""

        for attempt in range(self.max_retries + 1):
            result.n_retries = attempt

            # 1) build prompt and call Claude
            user_prompt, used_ex = self._build_user_prompt(question, plan, schema, retry_hint)
            if attempt == 0:
                result.exemplars_used = used_ex

            raw = self._call_claude(user_prompt)
            sql = self._extract_sql(raw)
            if attempt == 0:
                result.sql = sql

            # 2) PRE-EXECUTION TABLE-EXISTENCE CHECK (slide 4)
            ok, missing = check_tables_exist(sql, schema)
            if not ok:
                err_msg = (
                    f"pre-check: hallucinated tables {missing}. "
                    f"Valid table names: {[t.name for t in schema.tables]}. "
                    "Use ONLY tables that appear in the schema, with their fq_name."
                )
                result.attempts.append({
                    "attempt": attempt + 1, "sql": sql,
                    "error": err_msg, "phase": "table_check"
                })
                retry_hint = err_msg
                continue

            # 3) execute
            df, err = db_manager.execute(sql)
            if err is None:
                # success
                result.final_sql = sql
                result.df = df
                result.n_rows = len(df) if df is not None else 0
                result.error = None
                result.attempts.append({
                    "attempt": attempt + 1, "sql": sql,
                    "error": None, "phase": "executed", "n_rows": result.n_rows,
                })
                return result

            # error — classify and retry
            cls = classify_sql_error(err)
            err_msg = f"BigQuery error ({cls}): {err}"
            result.attempts.append({
                "attempt": attempt + 1, "sql": sql,
                "error": err_msg, "phase": "executed"
            })
            retry_hint = err_msg

        # exhausted retries
        result.final_sql = result.attempts[-1]["sql"] if result.attempts else ""
        result.error = retry_hint or "Unknown SQL error after retries."
        return result
