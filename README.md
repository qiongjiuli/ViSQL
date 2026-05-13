# ViSQL — A Vision-Augmented Autonomous Data Scientist

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Educational](https://img.shields.io/badge/license-educational-orange.svg)](#license)
[![Course: EECS 6895](https://img.shields.io/badge/course-EECS%206895-9cf.svg)](#)

**EECS 6895 · Advanced Big Data & AI · Columbia University · Spring 2026**
**Author:** Leah Li (`ql2481@columbia.edu`)

ViSQL is a six-stage hybrid pipeline that takes a natural-language analytical question and, optionally, a reference chart screenshot, and produces an **executed SQL query**, a **stylistically-matched chart**, and a **grounded analytical report** — end to end. It combines locally-served 4-bit quantized Llama models (router, vision, SQLCoder baseline) with Claude Sonnet 4.5 (planner, SQL agent, report writer), under three architectural guardrails designed to suppress the most damaging LLM failure modes in production analytics.

📄 **[Project report (PDF)](docs/visql_report.pdf)** ·  🎬 **[Demo video](#)** ·  🎨 **[Presentation slides (PDF)](docs/visql_slides.pdf)**

---

## Table of contents

- [What ViSQL does](#what-visql-does)
- [Architecture](#architecture)
- [Three architectural decisions](#three-architectural-decisions)
- [Headline results](#headline-results)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Quick start (notebook)](#quick-start-notebook)
- [Programmatic usage](#programmatic-usage)
- [Reproducing the evaluations](#reproducing-the-evaluations)
- [The demo UI](#the-demo-ui)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What ViSQL does

Production analysts spend most of their time on three pieces of plumbing that have little to do with the question they were asked: looking up schemas and writing dialect-correct SQL, reshaping the result into a stakeholder-ready chart, and translating findings into a narrative report. Each step is mechanical and error-prone — and a strong candidate for automation.

LLMs are individually competent at each piece, but **no public system stitches the whole loop together with the disciplines a senior analyst would impose**: matching a stakeholder's preferred visual style, refusing to make causal claims on observational data, and refusing to fabricate findings when a query returns nothing.

ViSQL closes that loop with:

1. **An end-to-end agentic pipeline** that routes a question, links it to a live BigQuery schema, plans it with chain-of-thought, generates and self-corrects SQL, dispatches one of five analysis branches (chart, dashboard, A/B test, ML modeling, raw extraction), and writes a grounded report.
2. **Three architectural guardrails** aimed at the most common LLM-analyst failure modes (described below).
3. **A multimodal style-imitation subsystem** — a vision-language model extracts a structured `StyleSpec` from a user-supplied reference chart; matplotlib then renders the data in that style. *Vision describes; matplotlib executes.*
4. **Four evaluation suites** covering the router (macro-F1), SQL generator (Spider exec-accuracy), vision style imitation (CIELAB ΔE self-consistency), and report writer (LLM-as-judge).

---

## Architecture

```
       ┌─────────────────────────────── CORE PIPELINE ────────────────────────────────┐
       │                                                                              │
NL ───▶│  1. Router      ─▶ 2. Schema link ─▶ 3. CoT planner                          │
       │  Llama-3.1-8B      MiniLM + FAISS    Claude Sonnet 4.5                       │
       │  + A/B gate        + live BQ         <thinking> + <plan>                     │
       │       │                  │                  │                                │
       │       ▼                  ▼                  ▼                                │
       │  6. Report  ◀── 5. Analysis  ◀──── 4. SQL agent                              │
       │  Claude 4.5    5 branches          Claude Sonnet 4.5                         │
       │  + empty-      scipy/sklearn       few-shot Spider                           │
       │    guard       /torch              + retry ≤ 3×                              │
       │                                                                              │
       └──────────────────────────────────────────────────────────────────────────────┘
                                                            ▲
                       Chart + Report + Trace               │ live schema, exec SQL
                              ▼                             │
                       ┌──────────────┐               ┌──────────────┐
ref chart (opt.) ────▶ │   Vision     │ ─StyleSpec─▶  │   BigQuery   │
                       │ Llama-3.2-V  │               │ SEC/theLook/ │
                       └──────────────┘               │      GA      │
                                                      └──────────────┘
```

**Hybrid local + API design.** Cheap, frequently-called components (5-way task router, style extraction) run on locally-quantized Llama models. Reasoning-heavy components (CoT planning, SQL generation with retry, report writing) call Claude Sonnet 4.5. The split is empirical: each model is placed where it earns its keep.

See `docs/visql_report.pdf` §3 for the full Figure 1.

---

## Three architectural decisions

These three decisions distinguish ViSQL from a vanilla agentic pipeline.

### 1. Strict A/B test gate &nbsp;·&nbsp; `visql/router.py::TaskRouter.apply_gate`

The `ab_test` route is the *only* branch that produces causal language in the report. We therefore impose a redundant filter on top of the LLM's classification: `ab_test` is allowed only if the question contains explicit experiment-language (*experiment, treatment, control, variant, lift, randomized,* etc.) **or** the schema contains a variant-assignment column. Otherwise the request is demoted to `single_chart`, where the report writer is forbidden from causal claims.

This rule prevents the most common analyst-LLM failure mode: a user asks *"is conversion different between mobile and desktop?"*, the system runs a chi-squared, and concludes *"mobile **caused** higher conversion."* The gate makes that report unreachable by construction.

### 2. Live schema introspection &nbsp;·&nbsp; `visql/schemas.py::fetch_bq_schema`, `check_tables_exist`

Schemas are fetched at runtime from `INFORMATION_SCHEMA.COLUMNS` and enriched with one sample row per table. This eliminates schema drift between a fixed snapshot and the production database. A pre-execution table-existence check intercepts hallucinated table names before paying for the BigQuery round-trip and surfaces a typed error back to the agent's retry loop.

### 3. Empty-data guard &nbsp;·&nbsp; `visql/analysis.py::ReportWriter.write`

When SQL returns zero rows, the report writer is bypassed entirely; instead the system emits the executed SQL, a diagnostic explanation, and a suggestion to inspect filters. In early iterations we repeatedly observed the LLM fabricating findings *"in the spirit of"* the question when it had no data — a high-stakes failure mode in any production setting. The guard makes that mode unreachable.

---

## Headline results

| Suite | Metric | Value | Source |
|---|---|---|---|
| Router | macro-F1 (60 hand-labeled questions) | **0.92** | `evals/router_eval.py` |
| Router (A/B gate) | adversarial-probe accuracy (10 probes) | **8/10** | `evals/router_eval.py` |
| SQL on Spider | exec-accuracy (200-ex Spider 1.0 dev) | **0.74** | `python -m evals.sql_eval` |
| SQL on Spider | retry-rate · recovery-rate | **17% · 55%** | `python -m evals.sql_eval` |
| SQL on Spider | first-attempt → final (marginal lift) | **0.68 → 0.74 (+6 pp)** | `python -m evals.sql_eval` |
| Style imitation | mean CIELAB ΔE-76 (12 refs) | **7.4** (below JND 10) | `evals/style_eval.py` |
| Style imitation | chart-type-hint accuracy | **0.83** | `evals/style_eval.py` |
| Reports (LLM-judge) | mean overall (6 q × 4 axes) | **4.1 / 5** | `evals/report_eval.py` |
| Reports | coverage of expected findings | **0.78** | `evals/report_eval.py` |

Numbers above are also discussed in `docs/visql_report.pdf` §5 and on slide 7–8 of the presentation deck.

---

## Repository structure

```
visql_v2/
├── README.md                          ← you are here
├── requirements.txt                   ← pinned dependencies
├── visql/                             ← CORE PIPELINE (importable as `import visql`)
│   ├── __init__.py                      package entrypoint; re-exports build_pipeline, ViSQLPipeline
│   ├── config.py                        model names, paths, hyperparameters (single source of truth)
│   ├── schemas.py                       DatabaseSchema dataclass + live BQ introspection + table-existence check
│   ├── llama_runtime.py                 LlamaText, LlamaVision, SQLCoder wrappers (4-bit, bitsandbytes)
│   ├── retrievers.py                    SchemaEmbedder + Spider ExemplarRetriever (sentence-transformers + FAISS)
│   ├── router.py                        Stage 1 — 5-way task classifier + STRICT A/B GATE
│   ├── planner.py                       Stage 3 — CoT planner with <thinking>/<plan> blocks (Claude)
│   ├── sql_agent.py                     Stage 4 — Claude SQL agent w/ few-shot, error-classifier, retry ≤ 3×
│   ├── analysis.py                      Stage 5 (5 branches) + Stage 6 (ReportWriter w/ empty-data guard)
│   ├── vision.py                        Multimodal — StyleSpec JSON extraction (Llama-Vision)
│   ├── renderer.py                      matplotlib renderer + StyleSpec application
│   ├── pipeline.py                      ViSQLPipeline orchestrator; the `build_pipeline()` factory
│   └── lora_train.py                    SQLCoder-7B-2 LoRA fine-tune (rank-16, attention+FFN, QLoRA-style)
├── evals/                             ← EVALUATION SUITES (each runnable standalone)
│   ├── router_eval.py                   60 hand-labeled questions + 10 adversarial probes → macro-F1 (0.92)
│   ├── sql_eval.py                      Spider 1.0 dev exec-acc + retry-rate + recovery-rate breakdown
│   ├── style_eval.py                    Self-consistency: extract → render → re-extract → CIELAB ΔE (7.4)
│   └── report_eval.py                   LLM-as-judge across 4 axes on 6 reference questions (4.1/5)
├── ui/                                ← STREAMLIT DEMO UI
│   ├── app.py                           Streamlit frontend with 7 tabs (Question, Route, Plan, SQL, Result, Chart, Report)
│   ├── backend.py                       Flask backend, in-process — shares the pipeline object (no IPC overhead)
│   └── launcher.py                      one-call helper: backend + Streamlit + ngrok tunnel
├── notebooks/
│   └── ViSQL_v2_Demo.ipynb              END-TO-END NOTEBOOK — installs deps, builds pipeline, runs 4 demos, runs all 4 evals, launches the UI
├── data/                              ← place reference chart screenshots here
│   └── style_refs/                      e.g., dark_minimalist.png, ft_style.png, …
└── docs/
    ├── visql_report.pdf                 6-page ACM SIGPLAN report
    └── visql_slides.pdf                 10-slide presentation deck
```

The `visql/` package is **importable as a library**; the `ui/` and `notebooks/` subfolders are clients of it. Each `evals/*.py` file is also runnable as `python -m evals.<name>` for reproducibility.

---

## Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | `requirements.txt` is pinned to ≥ 3.10 |
| **NVIDIA GPU (≥ 24 GB)** | Single A100-40GB is the reference config. T4 (16 GB) works for inference but not LoRA training. CPU-only works for the UI shell but local Llama inference will be too slow. |
| **Anthropic API key** | for Claude Sonnet 4.5 (planner, SQL agent, report writer) |
| **GCP project with BigQuery enabled** | three public datasets are queried directly |
| **ngrok auth token** *(optional)* | only needed if you want to expose the UI from Colab |

### Install

```bash
git clone <this-repo> visql_v2 && cd visql_v2
pip install -r requirements.txt
```

### Configure credentials

Set as environment variables (or Colab secrets, or a `.env` file):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GCP_PROJECT=your-gcp-project-id
export NGROK_TOKEN=...   # only for the public-URL demo
```

For BigQuery auth in a notebook, `from google.colab import auth; auth.authenticate_user()` works in Colab; otherwise use a service-account JSON via `GOOGLE_APPLICATION_CREDENTIALS`.

---

## Quick start (notebook)

The fastest path: open `notebooks/ViSQL_v2_Demo.ipynb` in Colab or Jupyter and run top-to-bottom. The notebook is organized into 16 sections mapping to the slide deck — each section is self-contained, including the four eval suites and the LoRA fine-tune. Total runtime end-to-end is roughly 25–30 minutes on a single A100 (the LoRA fine-tune alone is ~2 hours and is the only optional step).

---

## Programmatic usage

ViSQL is also a library. The minimal call signature:

```python
from google.cloud import bigquery
import anthropic
from visql import build_pipeline
from visql.schemas import fetch_bq_schema

# 1) Bring your own BQ client + Claude client
bq     = bigquery.Client(project="my-gcp-project")
claude = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

# 2) Introspect the schemas you want ViSQL to see
schemas = {
    "thelook":     fetch_bq_schema(bq, "bigquery-public-data", "thelook_ecommerce", sample_rows=1),
    "ga":          fetch_bq_schema(bq, "bigquery-public-data", "google_analytics_sample", sample_rows=1),
    "sec":         fetch_bq_schema(bq, "bigquery-public-data", "sec_quarterly_financials", sample_rows=1),
}

# 3) Build the pipeline (loads local Llamas, builds FAISS index, etc.)
pipeline = build_pipeline(bq, claude, schemas=schemas)

# 4) Run a question
result = pipeline.run(
    question="Show me the top 10 product categories by revenue.",
    dataset_key="thelook",
)

# 5) Inspect the result
print(result.route)              # "single_chart"
print(result.plan.thinking)      # the CoT reasoning trace
print(result.final_sql)          # the executed SQL
result.dataframe.head()          # the data
result.figure.savefig("out.png") # the styled chart
print(result.report)             # the grounded narrative
```

### With a reference chart (style imitation)

```python
result = pipeline.run(
    question="Top 10 product categories by revenue.",
    dataset_key="thelook",
    reference_chart_path="data/style_refs/ft_style.png",  # any chart screenshot
)
result.style_spec     # the extracted StyleSpec JSON
result.figure         # rendered in that style
```

### Inspecting the self-correction trace

```python
result.sql_attempts   # list of {attempt, sql, error, error_class, n_rows}
result.n_retries      # number of retries that fired (0 if first-attempt success)
```

---

## Reproducing the evaluations

The four eval suites are all defined in `evals/`. The simplest way to run all four is the demo notebook (Sections 11–14), which constructs the `generate_fn` and reference data each suite needs and prints a summary block per suite.

The SQL eval also has a standalone CLI for convenience:

```bash
# Spider 1.0 dev exec-accuracy + retry-rate + recovery-rate + error-class distribution
# (matches the report's §5.2 and §5.5 numbers)
python -m evals.sql_eval --n 200 \
    --spider-dev data/spider/dev.json \
    --spider-db  data/spider/database \
    --out evals/results/sql_eval_summary.json
```

Output goes to `evals/results/` as a small JSON summary blob; per-example traces (gold SQL, predicted SQL, retry attempts, error classes) stay in memory and can be inspected from the notebook.

The other three suites are notebook-driven because they require fixtures (60 hand-labeled router questions, 6 reference report questions, 12 chart screenshots) that are easier to wire up inline than via command-line args. See the notebook's eval sections for one-cell invocations:

```python
# Router (60 hand-labeled + 10 adversarial gate probes -> 0.92 macro-F1)
from evals.router_eval import RouterEvaluator
RouterEvaluator(router).evaluate().pretty()

# Style imitation (12 references -> mean delta-E 7.4)
from evals.style_eval import StyleEvaluator
StyleEvaluator(vision, renderer).evaluate("data/style_refs/").pretty()

# Reports (6 questions x 4 axes -> 4.1/5)
from evals.report_eval import ReportEvaluator
ReportEvaluator(claude, pipeline).evaluate().pretty()
```

The Spider eval additionally requires the Spider 1.0 dev set; the notebook (Section 4) shows how to download and prep it.

---

## The demo UI

ViSQL ships with a Streamlit demo UI that surfaces every intermediate stage — route, plan, SQL, result, chart, report, plus reasoning trace and retry attempts — in seven tabs.

```python
from visql import build_pipeline
from ui.launcher import launch_ui

pipeline = build_pipeline(bq, claude, schemas=schemas)
handles  = launch_ui(pipeline, gcp_project="my-gcp-project", ngrok_token=NGROK_TOKEN)
print(handles["public_url"])  # opens the UI
```

The backend (Flask) and frontend (Streamlit) share the same in-process pipeline object — there is no IPC overhead and a question round-trips in ~3-8 s end to end on an A100.

**Three queries to try once the UI is up:**

1. `Show me the top 10 product categories by revenue.` → routed as `single_chart`, full styled output.
2. `For the checkout_redesign_2024 A/B test, did treatment lift conversion vs control?` → routed as `ab_test`, returns χ² + Wald CI + cited stats in the report.
3. `Is conversion significantly different between mobile and desktop users?` → LLM proposed `ab_test`, **A/B gate demotes to `single_chart`**, no causal language in report.

---

## Configuration reference

All tunable constants live in `visql/config.py`:

| Constant | Default | What it controls |
|---|---|---|
| `ROUTER_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | local router (4-bit) |
| `VISION_MODEL` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | StyleSpec extractor (4-bit) |
| `SQLCODER_MODEL` | `defog/sqlcoder-7b-2` | local SQL baseline (LoRA target) |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | planner, SQL agent, report writer |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | schema + exemplar embeddings |
| `TOP_K_SCHEMA` | `5` | top-k tables retrieved during schema linking |
| `TOP_K_EXEMPLARS` | `3` | top-k Spider examples injected into SQL prompt |
| `SQL_MAX_RETRIES` | `3` | self-correction loop budget |
| `BQ_TIMEOUT_S` | `30` | BigQuery query timeout |

The router's experiment-language whitelist (the A/B gate's primary signal) lives in `visql/router.py::EXPERIMENT_KEYWORDS` and is intentionally extensible.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ImportError: bitsandbytes` | Wrong CUDA. Try `pip install bitsandbytes --upgrade` or fall back to CPU by setting `LOAD_IN_4BIT=False` in `config.py` (slow). |
| `Forbidden: BigQuery API not enabled` | Enable BigQuery API in your GCP project; ensure your auth principal has `roles/bigquery.user` and `roles/bigquery.dataViewer` on the public datasets project. |
| `anthropic.AuthenticationError` | `ANTHROPIC_API_KEY` missing or invalid. |
| `OutOfMemoryError` loading Llama-3.2-Vision | The vision model needs ~7 GB at 4-bit. Free GPU memory by closing other notebook kernels, or skip vision (`pipeline.run(..., reference_chart_path=None)`). |
| Streamlit UI shows `connection refused` | Backend Flask server didn't start. Re-run `launch_ui(...)` and check the `handles["backend_log"]`. |
| Spider eval can't find the database files | Section 4 of the demo notebook downloads `spider.zip` (~95 MB) and points `SPIDER_DB_ROOT` at `data/spider/database/`. |
| `ngrok` tunnel times out | Free-tier ngrok rotates URLs. Re-run `launch_ui(...)` to get a new one, or pass a paid `NGROK_TOKEN`. |

---

## License

Educational use only. Course project for EECS 6895 (Columbia University, Spring 2026).

Datasets used are public:

- **Spider 1.0** — Yu et al., EMNLP 2018. CC BY-SA 4.0.
- **theLook E-commerce, Google Analytics sample, SEC Quarterly Financials** — Google Cloud Public Datasets.

Model weights are accessed via their respective providers (Anthropic API for Claude; Meta Llama license for Llama-3.x; Defog AI license for SQLCoder).
