
# import os
# import streamlit as st
# import pandas as pd

# from src.core import generate_bundle, score_student_response

# st.set_page_config(page_title="Lab Match (Vertical Slice)", layout="wide")

# # Determinism on provided sample: no RNG used; embeddings are deterministic for a fixed model+text.
# # Optional secrets (not required here). Example:
# # OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# st.title("🔬 Lab Match — Vertical Slice")
# st.caption("Type your interests and see the closest PI profiles from the sample dataset.")

# with st.sidebar:
#     st.header("Settings")
#     csv_path = st.text_input("Dataset path", "data/Prototype Dataset.csv")
#     k = st.slider("Top‑K", min_value=1, max_value=10, value=5, step=1)
#     #min_sim = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
#     #model_name = st.text_input("Embedding model (sentence-transformers)", "sentence-transformers/all-MiniLM-L6-v2")
#     #show_snippets = st.checkbox("Show text snippets", value=True)
# min_sim = 0.0
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# show_snippets = False

# @st.cache_resource(show_spinner=False)
# def _load_bundle(_csv_path: str, _model_name: str):
#     return generate_bundle(_csv_path, model_name=_model_name)

# with st.spinner("Loading model and building index… (cached)"):
#     bundle = _load_bundle(csv_path, model_name)

# st.success("Index ready.", icon="✅")

# default_writeup = (
#     "Interested in polymer electronics, conjugated backbones, "
#     "Prefer strong mentorship for industry‑oriented careers."
# )
# writeup = st.text_area("Your interests / background", value=default_writeup, height=120)

# if st.button("Find matches", type="primary"):
#     with st.spinner("Scoring…"):
#         table = score_student_response(writeup, bundle, k=k, min_similarity=min_sim)
#     if len(table) == 0:
#         st.warning("No results met the minimum similarity. Try lowering the threshold.")
#     else:
#         st.subheader("Results (double click snippet to read more)")
#         st.dataframe(table, use_container_width=True, hide_index=True)
#         st.bar_chart(table.set_index("name")["similarity"])
#         if show_snippets:
#             with st.expander("Snippets"):
#                 for _, row in table.iterrows():
#                     st.markdown(f"**{row['name']}** — sim={row['similarity']:.3f}\n\n> {row['snippet']}")

# st.markdown("---")
# st.caption("• Runs locally via `streamlit run streamlit_app.py`. • No API keys required.")
import streamlit as st
import pandas as pd

from src.core import generate_bundle, score_student_response

st.set_page_config(page_title="Lab Match (Vertical Slice)", layout="wide")

# Fixed model (no ability to change in UI)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BUNDLE_SCHEMA_VERSION = "v2-email-mapped"

st.title("🔬 Lab Match — Vertical Slice")
st.caption("Paste your interests and aspirations; see the closest PI profiles from the example dataset.")

# with st.sidebar:
#     st.header("Settings")
#     csv_path = st.text_input("Dataset path", "data/Synthetic_Dataset.csv")
#     k = st.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
#     min_sim = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# @st.cache_resource(show_spinner=False)
# def _load_bundle(_csv_path: str):
#     # Use fixed model name (no user control)
#     return generate_bundle(_csv_path, model_name=MODEL_NAME)

# with st.spinner("Loading model and building index… (cached)"):
#     bundle = _load_bundle(csv_path)
# st.success("Index ready.", icon="✅")
@st.cache_resource(show_spinner=False)
def _load_bundle(_csv_path: str, _schema_version: str):
    # schema_version is unused but part of the cache key
    from src.core import generate_bundle
    return generate_bundle(_csv_path, model_name=MODEL_NAME)

with st.sidebar:
    st.header("Settings")
    csv_path = st.text_input("Dataset path", "data/Synthetic_Dataset.csv")
    k = st.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
    min_sim = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    # if st.button("🔁 Rebuild index (clear cache)"):
    #     _load_bundle.clear()  # clears @st.cache_resource
    #     st.experimental_rerun()

with st.spinner("Loading model and building index… (cached)"):
    bundle = _load_bundle(csv_path, BUNDLE_SCHEMA_VERSION)
st.success("Index ready.", icon="✅")
###
default_writeup = (
    "Interested in polymer electronics, conjugated backbones, "
    "SEC/GPC in CHCl3, and strong mentorship for industry-oriented careers."
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
        display_cols = ["rank", "name", "department", "email", "similarity"]
        display_table = table[display_cols].copy()
        display_table["similarity"] = display_table["similarity"].map(lambda x: f"{x:.3f}")

        st.dataframe(display_table, use_container_width=True, hide_index=True)

        # Below the table: snippets (expanded to 500 chars, already provided by core)
        st.markdown("### Profile snippets (≤500 chars each)")
        for _, row in table.iterrows():
            st.markdown(f"**{row['name']}** — {row['department']}  \n"
                        f"*{row['email']}*  \n"
                        f"> {row['snippet']}")
