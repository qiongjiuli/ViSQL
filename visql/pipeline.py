"""Pipeline orchestrator — wires Stages 1-6 together.

This is the box-by-box implementation of the architecture diagram (slide 3):

    Inputs → [1] Router → [2] Schema link → [3] CoT planner →
             [4] SQL agent → [5] Analysis → [6] Report → Outputs

With the Vision module + Renderer running as a parallel subsystem when
the user provides a reference chart.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import time

import pandas as pd

from .schemas    import DatabaseSchema, DatabaseManager
from .router     import TaskRouter, RouterDecision
from .planner    import CoTPlanner, QueryPlan
from .sql_agent  import SQLAgent, SQLGenResult
from .vision     import VisionModule, StyleSpec
from .renderer   import render_chart, render_dashboard
from .analysis   import (
    ab_test, fit_linear, fit_tree, fit_nn,
    ReportWriter, ABTestResult, ModelResult,
)
from .retrievers import SchemaEmbedder

# ════════════════════════════════════════════════════════════════════
# RESULT
# ════════════════════════════════════════════════════════════════════
@dataclass
class PipelineResult:
    question: str = ""
    db_id: str = ""
    route: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    final_sql: str = ""
    sql_attempts: list[dict] = field(default_factory=list)
    n_retries: int = 0
    result_df: Optional[pd.DataFrame] = None
    n_rows: int = 0
    branch_output: dict = field(default_factory=dict)
    chart_paths: list[str] = field(default_factory=list)
    style_spec: dict = field(default_factory=dict)
    report: str = ""
    error: Optional[str] = None
    timing: dict = field(default_factory=dict)

# ════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════
class ViSQLPipeline:
    """Wires stages 1-6 + the multimodal subsystem."""

    def __init__(self,
                 router: TaskRouter,
                 planner: CoTPlanner,
                 sql_agent: SQLAgent,
                 report_writer: ReportWriter,
                 vision: Optional[VisionModule] = None,
                 schemas: Optional[dict[str, DatabaseSchema]] = None,
                 bq_client=None):
        self.router = router
        self.planner = planner
        self.sql_agent = sql_agent
        self.report_writer = report_writer
        self.vision = vision
        self.schemas = schemas or {}
        self.bq = bq_client
        self._embedders: dict[str, SchemaEmbedder] = {}

    def get_schema(self, db_id: str) -> DatabaseSchema:
        if db_id not in self.schemas:
            raise KeyError(f"Unknown db_id: {db_id}. Known: {list(self.schemas)}")
        return self.schemas[db_id]

    def _get_embedder(self, schema: DatabaseSchema) -> SchemaEmbedder:
        if schema.db_id not in self._embedders:
            self._embedders[schema.db_id] = SchemaEmbedder(schema)
        return self._embedders[schema.db_id]

    # ── Stage 5 dispatcher ──────────────────────────────────────
    def _run_analysis_branch(self, route: str, df: pd.DataFrame, plan: QueryPlan) -> dict:
        """Stage 5 — dispatch to the appropriate analysis branch."""
        out: dict[str, Any] = {"route": route}
        if df is None or len(df) == 0:
            return out

        if route == "ab_test":
            res = ab_test(df)
            out["ab_test"] = res.to_dict()

        elif route == "ml_modeling":
            # try to infer target & features from the dataframe
            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(exclude="number").columns.tolist()
            target = None
            for cand in ("is_returned", "converted", "churn", "label", "target", "y"):
                if cand in df.columns:
                    target = cand
                    break
            if target is None and num_cols:
                target = num_cols[-1]  # use last numeric col as a fallback target
            if target is None:
                out["ml_modeling"] = {"error": "could not infer target"}
                return out
            features = [c for c in df.columns if c != target][:8]
            try:
                lin = fit_linear(df, target, features)
                tree = fit_tree(df, target, features)
                nn = fit_nn(df, target, features, epochs=20)
                out["ml_modeling"] = {
                    "linear": lin.summary,
                    "tree":   tree.summary,
                    "nn":     nn.summary,
                }
            except Exception as e:
                out["ml_modeling"] = {"error": str(e)}

        elif route == "single_chart":
            out["single_chart"] = {"chart_hint": plan.chart_hint, "n_rows": len(df)}

        elif route == "dashboard":
            out["dashboard"] = {"n_rows": len(df)}

        elif route == "sql_only":
            out["sql_only"] = {"n_rows": len(df)}

        return out

    # ── Stage Vision (parallel subsystem) ───────────────────────
    def _maybe_extract_style(self, reference_image) -> Optional[StyleSpec]:
        if reference_image is None or self.vision is None:
            return None
        try:
            return self.vision.extract_style(reference_image)
        except Exception as e:
            print(f"[vision] extraction failed: {e}")
            return None

    # ── Main run ─────────────────────────────────────────────────
    def run(self,
            question: str,
            db_id: str,
            reference_image=None,
            verbose: bool = False) -> PipelineResult:
        result = PipelineResult(question=question, db_id=db_id)
        timing: dict[str, float] = {}
        schema = self.get_schema(db_id)

        # ── Vision subsystem (in parallel; we just sequence here) ──
        t0 = time.time()
        style_spec = self._maybe_extract_style(reference_image)
        if style_spec is not None:
            result.style_spec = style_spec.to_dict()
        timing["vision"] = time.time() - t0

        # ── STAGE 1 — Router ───────────────────────────────────
        t0 = time.time()
        decision: RouterDecision = self.router.route(question, schema)
        timing["router"] = time.time() - t0
        result.route = {
            "label": decision.label,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "gated_from": decision.gated_from,
        }
        if verbose:
            print(f"[1/6] route → {decision.label} (conf={decision.confidence:.2f})"
                  + (f"  [GATED from {decision.gated_from}]" if decision.gated_from else ""))

        # ── STAGE 2 — Schema linking ──────────────────────────
        t0 = time.time()
        try:
            embedder = self._get_embedder(schema)
            linked_schema = embedder.link_to_schema(question, k=6)
        except Exception as e:
            if verbose:
                print(f"[2/6] schema linking failed ({e}); using full schema")
            linked_schema = schema
        timing["schema_link"] = time.time() - t0
        if verbose:
            print(f"[2/6] schema linked ({len(linked_schema.tables)} tables)")

        # ── STAGE 3 — CoT planner ────────────────────────────
        t0 = time.time()
        plan = self.planner.plan(question, linked_schema, route_label=decision.label)
        timing["planner"] = time.time() - t0
        result.plan = plan.to_dict()
        if verbose:
            print(f"[3/6] plan: {plan.complexity} / {plan.chart_hint}")

        # ── STAGE 4 — SQL agent ──────────────────────────────
        t0 = time.time()
        db_mgr = DatabaseManager(linked_schema, bq_client=self.bq)
        sql_res: SQLGenResult = self.sql_agent.generate_and_execute(
            question, plan, linked_schema, db_mgr
        )
        timing["sql"] = time.time() - t0
        result.final_sql = sql_res.final_sql or sql_res.sql
        result.sql_attempts = sql_res.attempts
        result.n_retries = sql_res.n_retries
        result.result_df = sql_res.df
        result.n_rows = sql_res.n_rows
        if sql_res.error:
            result.error = sql_res.error
        if verbose:
            err_preview = (sql_res.error[:120] + "...") if sql_res.error else None
            print(f"[4/6] SQL ({sql_res.n_retries} retries) "
                  + (f"→ err: {err_preview}" if err_preview else f"→ {sql_res.n_rows} rows"))

        # ── STAGE 5 — Analysis dispatch ──────────────────────
        t0 = time.time()
        branch_output = self._run_analysis_branch(decision.label, sql_res.df, plan)
        timing["analysis"] = time.time() - t0
        result.branch_output = branch_output

        # Render chart(s)
        chart_paths = []
        if sql_res.df is not None and len(sql_res.df) > 0:
            if decision.label == "dashboard":
                p = render_dashboard([sql_res.df], titles=[plan.intent or "Result"],
                                     chart_types=[plan.chart_hint],
                                     style=style_spec or StyleSpec())
                if p: chart_paths.append(p)
            elif decision.label != "sql_only":
                p = render_chart(sql_res.df, chart_type=plan.chart_hint,
                                 title=plan.intent or "", style=style_spec or StyleSpec())
                if p: chart_paths.append(p)
        result.chart_paths = chart_paths
        if verbose:
            print(f"[5/6] branch={decision.label}, charts={len(chart_paths)}")

        # ── STAGE 6 — Report ────────────────────────────────
        t0 = time.time()
        # Hand the appropriate analysis result to the report writer
        ab_obj: Optional[ABTestResult] = None
        ml_obj: Optional[ModelResult] = None
        if "ab_test" in branch_output and isinstance(branch_output["ab_test"], dict):
            ab_obj = ABTestResult(**{k: v for k, v in branch_output["ab_test"].items()
                                      if k in ABTestResult.__annotations__})

        report = self.report_writer.write(
            question=question,
            df=sql_res.df,
            executed_sql=result.final_sql,
            plan=result.plan,
            ab_result=ab_obj,
            model_result=ml_obj,
        )
        timing["report"] = time.time() - t0
        result.report = report
        if verbose:
            print(f"[6/6] report ({len(report)} chars)")

        result.timing = timing
        return result

# ════════════════════════════════════════════════════════════════════
# CONVENIENCE BUILDER
# ════════════════════════════════════════════════════════════════════
def build_pipeline(bq_client,
                   anthropic_client,
                   schemas: Optional[dict] = None,
                   exemplars: Optional[list[dict]] = None,
                   load_vision: bool = True,
                   load_sqlcoder: bool = False) -> ViSQLPipeline:
    """One-call pipeline construction.

    Args:
        bq_client       : google.cloud.bigquery.Client (or None for offline tests)
        anthropic_client: anthropic.Anthropic instance
        schemas         : dict[db_id -> DatabaseSchema]; if None, uses defaults
        exemplars       : Spider exemplars for few-shot SQL; if None, no few-shot
        load_vision     : whether to load Llama-Vision (slow; skip for SQL-only demos)
        load_sqlcoder   : load SQLCoder for the LoRA baseline eval (not used in pipeline)
    """
    from .llama_runtime import LlamaText, LlamaVision
    from .retrievers   import ExemplarRetriever
    from .schemas      import default_schemas_live

    print("[build] loading Llama text (router)...")
    text_llm = LlamaText()

    print("[build] building TaskRouter...")
    router = TaskRouter(text_llm)

    print("[build] building CoTPlanner...")
    planner = CoTPlanner(anthropic_client)

    print("[build] building SQLAgent (with exemplars)..." if exemplars else "[build] building SQLAgent...")
    exemplar_retriever = ExemplarRetriever(exemplars) if exemplars else None
    sql_agent = SQLAgent(anthropic_client, exemplar_retriever=exemplar_retriever)

    print("[build] building ReportWriter...")
    report_writer = ReportWriter(anthropic_client)

    vision_module = None
    if load_vision:
        print("[build] loading Llama vision...")
        vision_llm = LlamaVision()
        vision_module = VisionModule(vision_llm)

    if schemas is None and bq_client is not None:
        schemas = default_schemas_live(bq_client)

    return ViSQLPipeline(
        router=router,
        planner=planner,
        sql_agent=sql_agent,
        report_writer=report_writer,
        vision=vision_module,
        schemas=schemas or {},
        bq_client=bq_client,
    )
