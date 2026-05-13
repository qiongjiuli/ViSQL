"""Flask backend for the Streamlit UI.

Runs in the *same* Python process as the notebook (in a daemon thread) so it
can share the already-loaded ViSQLPipeline (and its GBs of model weights).
The Streamlit subprocess hits this over HTTP at 127.0.0.1:8765.

Endpoints:
    GET  /health  -> { status, project, datasets }
    POST /run     -> { question, db_id, reference_image_b64? }
                  -> serialized PipelineResult (with charts as base64 PNGs)
"""
from __future__ import annotations
import base64
import io
import threading
from typing import Optional

from flask import Flask, request, jsonify
from PIL import Image

from visql.pipeline import ViSQLPipeline

# Module-level handles, set by run_backend()
_app: Optional[Flask] = None
_pipeline: Optional[ViSQLPipeline] = None
_user_project: str = "(unknown)"


def _b64_png(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def _serialize_result(r) -> dict:
    """PipelineResult -> JSON-friendly dict for the Streamlit client."""
    rows = []
    if r.result_df is not None:
        # cap at 500 rows for transport size
        rows = r.result_df.head(500).to_dict(orient="records")

    chart_pngs_b64 = []
    for cp in (r.chart_paths or []):
        b = _b64_png(cp)
        if b:
            chart_pngs_b64.append(b)

    return {
        "question":      r.question,
        "db_id":         r.db_id,
        "route":         r.route,
        "plan":          r.plan,
        "final_sql":     r.final_sql,
        "sql_attempts":  r.sql_attempts,
        "n_retries":     r.n_retries,
        "n_rows":        r.n_rows,
        "rows":          rows,
        "branch_output": r.branch_output,
        "chart_pngs_b64": chart_pngs_b64,
        "style_spec":    r.style_spec,
        "report":        r.report,
        "error":         r.error,
        "timing":        r.timing,
    }


def make_app(pipeline: ViSQLPipeline, user_project: str) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "project": user_project,
            "datasets": list(pipeline.schemas.keys()),
        })

    @app.post("/run")
    def run():
        data = request.get_json(force=True) or {}
        question = data.get("question", "").strip()
        db_id    = data.get("db_id", "thelook")
        if not question:
            return jsonify({"error": "empty question"}), 400

        ref_img = None
        b64 = data.get("reference_image_b64")
        if b64:
            try:
                ref_img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            except Exception as e:
                return jsonify({"error": f"bad image: {e}"}), 400

        try:
            result = pipeline.run(
                question=question,
                db_id=db_id,
                reference_image=ref_img,
                verbose=True,
            )
        except Exception as e:
            import traceback
            return jsonify({"error": f"{e}\n{traceback.format_exc()[:1500]}"}), 500

        return jsonify(_serialize_result(result))

    return app


def run_backend(pipeline: ViSQLPipeline,
                user_project: str = "(unknown)",
                host: str = "127.0.0.1",
                port: int = 8765) -> threading.Thread:
    """Launch the Flask app in a daemon thread. Returns the thread."""
    global _app, _pipeline, _user_project
    _pipeline = pipeline
    _user_project = user_project
    _app = make_app(pipeline, user_project)

    def _serve():
        # use_reloader=False is REQUIRED — reloader spawns a child process
        # which would re-import everything and lose the pipeline reference.
        _app.run(host=host, port=port, debug=False,
                 use_reloader=False, threaded=True)

    t = threading.Thread(target=_serve, daemon=True, name="visql-backend")
    t.start()
    print(f"[backend] Flask server listening on http://{host}:{port}")
    return t
