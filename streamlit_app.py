import streamlit as st
import pandas as pd

from src.core import generate_bundle, score_student_response

st.set_page_config(page_title="Lab Match (Vertical Slice)", layout="wide")

# Fixed model (no ability to change in UI)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BUNDLE_SCHEMA_VERSION = "v2-email-mapped"

st.title("🔬 Lab Match — Vertical Slice")
st.caption("Paste your interests and aspirations; see the closest PI profiles from the example dataset.")


@st.cache_resource(show_spinner=False)
def _load_bundle(_csv_path: str, _schema_version: str):
    # schema_version is unused but part of the cache key
    from src.core import generate_bundle
    return generate_bundle(_csv_path, model_name=MODEL_NAME)

with st.sidebar:
    st.header("Settings")
    csv_path = st.text_input("Dataset path", "data/Synthetic_Dataset.csv")
    k = st.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
    #min_sim = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
min_sim =0.0  # Fixed at 0.0 for this vertical slice

with st.spinner("Loading model and building index… (cached)"):
    bundle = _load_bundle(csv_path, BUNDLE_SCHEMA_VERSION)
st.success("Index ready.", icon="✅")

default_writeup = (
    "Example: Interested in polymer electronics, conjugated backbones, "
    "Prefer strong mentorship and set on industry-oriented careers."
)
writeup = st.text_area("Your interests / background", value=default_writeup, height=120)

if st.button("Find matches", type="primary"):
    with st.spinner("Scoring…"):
        table = score_student_response(writeup, bundle, k=k, min_similarity=min_sim)

    if len(table) == 0:
        st.warning("No results met the minimum similarity. Try lowering the threshold.")
    else:
        st.subheader("Results")

        # Remove ID and snippet from the display table; include email + department
        display_cols = ["rank", "similarity", "name", "department", "email"]
        display_table = table[display_cols].copy()
        display_table["similarity"] = display_table["similarity"].map(lambda x: f"{x:.3f}")

        st.dataframe(display_table, use_container_width=True, hide_index=True)

        # Below the table: snippets (expanded to 500 chars, already provided by core)
        st.markdown("### Profile snippets (≤500 chars each)")
        for _, row in table.iterrows():
            st.markdown(f"**{row['name']}** — {row['department']}  \n"
                        f"*{row['email']}*  \n"
                        f"> {row['snippet']}")
