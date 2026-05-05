# ViSQL v2 — Vision-Augmented Autonomous Data Scientist

EECS 6895 · Columbia · Leah Li (ql2481@columbia.edu)

A 6-stage pipeline that takes a natural-language question and (optionally) a reference chart screenshot, and produces a SQL query, a styled chart, and a grounded analytical report. Hybrid Llama-local + Claude-API design.

## Quick start

```bash
git clone <this-repo> /content/visql_v2
cd /content/visql_v2
pip install -r requirements.txt

# Set these as env vars or Colab secrets:
export ANTHROPIC_API_KEY=sk-ant-...
export GCP_PROJECT=your-bq-project
export NGROK_TOKEN=...   # for the demo UI
```

Then open `notebooks/ViSQL_v2_Demo.ipynb` and run top-to-bottom.

## Project map

```
visql_v2/
├── visql/                 # core pipeline (importable as `visql`)
│   ├── config.py          # model names, paths, hyperparams
│   ├── schemas.py         # DatabaseSchema + live BQ introspection + table-existence check
│   ├── llama_runtime.py   # LlamaText / LlamaVision / SQLCoder wrappers
│   ├── retrievers.py      # SchemaEmbedder + Spider ExemplarRetriever (FAISS)
│   ├── router.py          # Stage 1 — 5-way classifier + STRICT A/B GATE
│   ├── planner.py         # Stage 3 — CoT planner with <thinking> trace (Claude)
│   ├── sql_agent.py       # Stage 4 — SQL agent with table-check + retry ≤3× (Claude)
│   ├── analysis.py        # Stage 5 — A/B + ML branches; Stage 6 — ReportWriter w/ empty-data guard
│   ├── vision.py          # Multimodal — StyleSpec extraction (Llama-Vision)
│   ├── renderer.py        # matplotlib + StyleSpec rendering
│   ├── pipeline.py        # ViSQLPipeline orchestrator (build_pipeline helper)
│   └── lora_train.py      # SQLCoder LoRA fine-tune (rank-16, attention+FFN)
├── evals/                 # evaluation suites (slide 10)
│   ├── router_eval.py     # 60 hand-labeled + 10 adversarial (0.92 macro-F1)
│   ├── style_eval.py      # CIELAB ΔE-76 self-consistency (ΔE 7.4)
│   ├── report_eval.py     # LLM-as-judge with 4-axis rubric (4.1/5)
│   └── sql_eval.py        # Spider 1.0 dev exec accuracy (0.74)
├── ui/                    # Streamlit demo UI
│   ├── app.py             # Streamlit frontend (7 tabs)
│   ├── backend.py         # Flask backend, in-process — shares the pipeline
│   └── launcher.py        # one-call: backend + Streamlit + ngrok
├── notebooks/
│   └── ViSQL_v2_Demo.ipynb  # end-to-end walkthrough
├── data/                  # place reference chart images here (style_refs/)
└── requirements.txt
```

## Architecture (slide 3)

```
Question + (optional) reference chart
            │
            ▼
   ┌────────────────────────────────────────────────────────────────┐
   │                       CORE PIPELINE                             │
   │  [1] Router        ─── Llama-3.1-8B (local, 4-bit)              │
   │  [2] Schema link   ─── MiniLM-L6-v2 + FAISS top-K (local)       │
   │  [3] CoT planner   ─── Claude Sonnet 4.5 (API) — emits <thinking> │
   │  [4] SQL agent     ─── Claude (API) + few-shot Spider + retry ≤3× │
   │  [5] Analysis      ─── 5 branches dispatched (scipy / sklearn / torch) │
   │  [6] Report writer ─── Claude (API) + empty-data guard           │
   └────────────────────────────────────────────────────────────────┘
            │                                   ▲
            ▼                                   │ live schema, execute SQL
     Chart + Report + Trace                   BigQuery
                                                ▲
                                                │
                                Vision Module ──┘
                                Llama-3.2-11B-Vision
                                StyleSpec extraction
```

## Three architectural decisions (slide 4)

1. **Strict A/B test gate.** `ab_test` route only fires if the question contains experiment-language OR the schema has a variant column. Demoting to `single_chart` otherwise. See `visql/router.py::TaskRouter.apply_gate`.
2. **Live schema introspection.** Schemas pulled from `INFORMATION_SCHEMA.COLUMNS`, sample rows enriched in. Pre-execution `check_tables_exist()` catches hallucinated tables. See `visql/schemas.py::fetch_bq_schema`, `check_tables_exist`.
3. **Empty-data guard.** If SQL returns 0 rows, `ReportWriter` refuses to narrate; returns the executed SQL + diagnostic. See `visql/analysis.py::ReportWriter.write`.

## Evaluation results (slide 10)

| Suite | Metric | Value | Where |
|---|---|---|---|
| Router | macro-F1 | 0.92 | `evals/router_eval.py` |
| Style imitation | mean ΔE-76 | 7.4 | `evals/style_eval.py` |
| Reports (LLM-judge) | mean overall | 4.1/5 | `evals/report_eval.py` |
| SQL on Spider | exec accuracy | 0.74 | `evals/sql_eval.py` |

## Demo

In a notebook:

```python
from visql import build_pipeline
from ui.launcher import launch_ui

pipeline = build_pipeline(bq, claude, schemas=schemas, exemplars=exemplars)
handles = launch_ui(pipeline, USER_PROJECT, ngrok_token=NGROK_TOKEN)
# click handles['public_url']
```

Then in the Streamlit UI:
1. Pick a dataset.
2. Ask: `Show me the top 10 product categories by revenue.` (single_chart)
3. Ask: `For the checkout_redesign_2024 A/B test, did treatment lift conversion vs control?` (ab_test)
4. Ask: `Is conversion significantly different between mobile and desktop users?` (gated → single_chart)

## License

Educational use only.
