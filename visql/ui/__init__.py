"""Streamlit demo UI for ViSQL v2.

The UI is a thin Streamlit front-end that talks to a Flask backend over
HTTP. The backend keeps the pipeline (and the loaded model weights) alive
across page reloads.

To launch in a notebook:
    from ui.launcher import launch_ui
    launch_ui(pipeline, schemas, USER_PROJECT, ngrok_token=...)
"""
