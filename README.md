
# Lab Match — Vertical Slice (Streamlit)

A Lablink demo that embeds PI/lab profiles and lets a student paste their interests and aspirations to find the closest matches.

## Quickstart

```bash
# 1) Clone/download this folder, then:
cd vertical_slice_app

# 2) (Recommended) create a venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install only what we use
pip install -r requirements.txt

# 4) Run locally
streamlit run streamlit_app.py
```

> The app loads `data/Prototype Dataset.csv` and builds a cosine-similarity index with `sentence-transformers/all-MiniLM-L6-v2`.  
> Results are deterministic for a fixed model + dataset.

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo (keep `streamlit_app.py` at the root).
2. In Streamlit Cloud: **New app → Select repo → Main file path: `streamlit_app.py`**.
3. No secrets are required for the default flow. If you later add keys, set them in **App → Settings → Secrets**.

## Structure

```
vertical_slice_app/
├─ data/
│  └─ Prototype Dataset.csv            # provided sample (no keys required)
├─ docs/
│  └─ screenshot.png                   # (optional) add your own screenshot
├─ src/
│  └─ core.py                          # refactored functions, minimal deps
├─ streamlit_app.py                    # simple UI importing src.core
├─ requirements.txt
└─ README.md
```

## Minimal dependencies

- `streamlit`, `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`  
- We **do not** hard‑code any keys. If you introduce APIs later, use Streamlit Secrets:

```toml
# .streamlit/secrets.toml (do NOT commit)
OPENAI_API_KEY = "sk-..."
```

And in code:

```python
import streamlit as st
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
```

## Sanity checks / pitfalls

- **Determinism:** With the bundled CSV and model name unchanged, results are deterministic.
- **Cold start:** First run downloads the sentence-transformers model (~90MB). Cached afterwards.
- **Model change:** If you change the model name, results will differ by design.
- **CSV schema:** If you bring your own CSV, keep columns like `pi_name`, `research_summary`, `keywords` etc.
- **Memory:** On Streamlit Cloud free tier, keep datasets modest (<10–20k rows) or precompute embeddings.

## Screenshots / GIFs

Add a screenshot (e.g., `docs/screenshot.png`) and embed it here:

![screenshot](docs/screenshot.png)
