"""One-call launcher: backend thread + Streamlit subprocess + ngrok tunnel.

Usage in a Colab/Jupyter notebook:

    from ui.launcher import launch_ui
    launch_ui(pipeline, USER_PROJECT, ngrok_token=NGROK_TOKEN)

This:
  1. Starts the Flask backend in a daemon thread (in-process — shares the pipeline).
  2. Launches `streamlit run ui/app.py` as a subprocess on port 8501.
  3. Opens an ngrok HTTPS tunnel pointing at 8501 and prints the public URL.
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

from .backend import run_backend


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    import socket
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.25)
    return False


def launch_ui(
    pipeline,
    user_project: str = "(unknown)",
    ngrok_token: Optional[str] = None,
    backend_port: int = 8765,
    streamlit_port: int = 8501,
    app_path: Optional[str] = None,
) -> dict:
    """Launch the full demo UI.

    Returns a dict with the public ngrok URL (if used) and process handles.
    """
    # 1) backend
    backend_thread = run_backend(pipeline, user_project=user_project, port=backend_port)
    if not _wait_for_port("127.0.0.1", backend_port, timeout=8):
        raise RuntimeError(f"Backend never came up on :{backend_port}")

    # 2) streamlit subprocess
    if app_path is None:
        app_path = str(Path(__file__).parent / "app.py")
    env = os.environ.copy()
    env["VISQL_BACKEND"] = f"http://127.0.0.1:{backend_port}"

    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(streamlit_port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
    ]
    print(f"[launcher] starting Streamlit: {' '.join(cmd)}")
    streamlit_proc = subprocess.Popen(cmd, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if not _wait_for_port("127.0.0.1", streamlit_port, timeout=20):
        streamlit_proc.terminate()
        raise RuntimeError(f"Streamlit never came up on :{streamlit_port}")
    print(f"[launcher] Streamlit ready on :{streamlit_port}")

    # 3) ngrok tunnel (Colab needs this; locally you could just use localhost)
    public_url = None
    if ngrok_token:
        try:
            from pyngrok import ngrok, conf
            conf.get_default().auth_token = ngrok_token
            # Kill any existing tunnels (avoid the "1 tunnel limit on free tier" error)
            for t in ngrok.get_tunnels():
                ngrok.disconnect(t.public_url)
            tunnel = ngrok.connect(streamlit_port, "http")
            public_url = tunnel.public_url.replace("http://", "https://")
            print(f"[launcher] PUBLIC URL: {public_url}")
        except Exception as e:
            print(f"[launcher] ngrok failed: {e}; UI is at http://127.0.0.1:{streamlit_port}")
    else:
        print(f"[launcher] no ngrok token; UI is at http://127.0.0.1:{streamlit_port}")

    return {
        "public_url": public_url,
        "local_url": f"http://127.0.0.1:{streamlit_port}",
        "backend_url": f"http://127.0.0.1:{backend_port}",
        "streamlit_proc": streamlit_proc,
        "backend_thread": backend_thread,
    }


def stop_ui(handles: dict) -> None:
    """Tear down the launched UI."""
    proc = handles.get("streamlit_proc")
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    if handles.get("public_url"):
        try:
            from pyngrok import ngrok
            for t in ngrok.get_tunnels():
                ngrok.disconnect(t.public_url)
            ngrok.kill()
        except Exception:
            pass
    print("[launcher] UI stopped")
