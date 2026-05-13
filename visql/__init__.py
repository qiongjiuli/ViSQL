"""ViSQL v2 — Vision-Augmented Autonomous Data Scientist."""
from .config        import (
    LLAMA_TEXT_MODEL, LLAMA_VISION_MODEL, SQL_BASE_MODEL, CLAUDE_MODEL,
    DEVICE, DTYPE, CACHE_DIR, TASK_LABELS,
)
from .schemas       import (
    DatabaseSchema, SchemaTable, SchemaColumn, DatabaseManager,
    fetch_bq_schema, default_schemas_live, check_tables_exist,
    create_synthetic_experiment_table, EXPERIMENT_SCHEMA,
)
from .llama_runtime import LlamaText, LlamaVision, SQLCoder
from .retrievers    import SchemaEmbedder, ExemplarRetriever, build_spider_exemplars
from .router        import TaskRouter, RouterDecision
from .planner       import CoTPlanner, QueryPlan
from .sql_agent     import SQLAgent, SQLGenResult, classify_sql_error
from .vision        import VisionModule, StyleSpec
from .renderer      import render_chart, render_dashboard
from .analysis      import (
    ab_test, fit_linear, fit_tree, fit_nn,
    ReportWriter, ABTestResult, ModelResult,
)
from .pipeline      import ViSQLPipeline, PipelineResult, build_pipeline

__all__ = [
    "LLAMA_TEXT_MODEL", "LLAMA_VISION_MODEL", "SQL_BASE_MODEL", "CLAUDE_MODEL",
    "DEVICE", "DTYPE", "CACHE_DIR", "TASK_LABELS",
    "DatabaseSchema", "SchemaTable", "SchemaColumn", "DatabaseManager",
    "fetch_bq_schema", "default_schemas_live", "check_tables_exist",
    "create_synthetic_experiment_table", "EXPERIMENT_SCHEMA",
    "LlamaText", "LlamaVision", "SQLCoder",
    "SchemaEmbedder", "ExemplarRetriever", "build_spider_exemplars",
    "TaskRouter", "RouterDecision",
    "CoTPlanner", "QueryPlan",
    "SQLAgent", "SQLGenResult", "classify_sql_error",
    "VisionModule", "StyleSpec",
    "render_chart", "render_dashboard",
    "ab_test", "fit_linear", "fit_tree", "fit_nn",
    "ReportWriter", "ABTestResult", "ModelResult",
    "ViSQLPipeline", "PipelineResult", "build_pipeline",
]
