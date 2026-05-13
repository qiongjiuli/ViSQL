# Reference chart screenshots

Drop chart screenshots here to drive the **style-imitation subsystem**. Any PNG/JPG works.

ViSQL's vision module (`visql/vision.py`) extracts a structured `StyleSpec` from the image — palette, gridlines, axis style, spine treatment, color mood, and a chart-type hint — which the renderer (`visql/renderer.py`) then applies to matplotlib.

The vision model never produces pixels; it only describes style. So adding a new reference image cannot introduce hallucinated data.

Suggested test set used in `evals/style_eval.py`: a 12-image mix of FT-style charts, Bloomberg dark-theme, Tableau defaults, Excel screenshots, hand-drawn axes, etc. The folder is intentionally left empty in the repo — bring your own.
