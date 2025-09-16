
"""
Core utilities for the vertical slice demo.
Refactored from the uploaded KG_vertical_slice.ipynb to a pure-Python module.
Dependencies: pandas, numpy, scikit-learn, sentence-transformers
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# ------------------------- Text prep -------------------------
CANDIDATE_TEXT_COLS = [
    "research_summary", "keywords", "mentorship_info",
    "publications", "publication_abstracts", "lab_overview",
    "social_media_posts", "scraped_excerpts", "awards"
]

def sanitize_text(s: str) -> str:
    """Remove URLs, emails, collapse whitespace; keep it deterministic."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"https?://\S+", " ", s)         # drop URLs
    s = re.sub(r"\S+@\S+\.\S+", " ", s)         # drop emails
    s = re.sub(r"\s+", " ", s).strip()          # collapse whitespace
    return s

def build_profile_text(row: pd.Series, cols: List[str]) -> str:
    """Concatenate non-empty fields from the provided columns."""
    parts = []
    for c in cols:
        if c in row and isinstance(row[c], str):
            t = sanitize_text(row[c])
            if t:
                parts.append(t)
    return " | ".join(parts)

# def load_profiles_csv_prototype(csv_path: str, key_text_cols: List[str] | None = None, min_chars: int = 20) -> pd.DataFrame:
#     """
#     Load CSV and prepare text for embedding.
#     Returns a reduced DataFrame with just id, name, text.
#     """
#     df = pd.read_csv(csv_path)
#     for c in ("pi_name","source_url"):
#         if c not in df.columns:
#             df[c] = ""
#     cols = [c for c in (key_text_cols or CANDIDATE_TEXT_COLS) if c in df.columns]
#     df["text"] = df.apply(lambda r: build_profile_text(r, cols), axis=1).astype(str).str.strip()
#     df = df[df["text"].str.len() >= min_chars].copy()

#     # deterministic ID using pi_name + source_url fallback
#     def _mkid(r):
#         basis = f"{r.get('pi_name','')}|{r.get('source_url','')}"
#         basis = basis.strip()
#         if basis:
#             return hashlib.md5(basis.encode("utf-8")).hexdigest()
#         return f"row-{int(r.name)}"
#     df["id"] = df.apply(_mkid, axis=1)
#     df["name"] = df.get("pi_name", "").astype(str).fillna("")
#     return df[["id","name","text"]].drop_duplicates("id").reset_index(drop=True)
def load_profiles_csv_prototype(csv_path: str, key_text_cols: List[str] | None = None, min_chars: int = 20) -> pd.DataFrame:
    """
    Load CSV and prepare text for embedding.
    Returns a reduced DataFrame with id, name, email, department, text.
    """
    df = pd.read_csv(csv_path)

    # Ensure always-present fields
    for c in ("pi_name", "source_url", "department"):
        if c not in df.columns:
            df[c] = ""

    # --- Email column detection (handles your 'contact_email') ---
    email_col_candidates = [
        "email", "contact_email", "pi_email", "contact", "e-mail", "Email", "E-mail"
    ]
    email_col = next((c for c in email_col_candidates if c in df.columns), None)
    if email_col is None:
        # Fall back to blank if no email-like column is present
        df["email"] = ""
    else:
        # Clean common NaN/None string artifacts
        df["email"] = (
            df[email_col]
            .astype(str)
            .replace({"nan": "", "NaN": "", "None": ""}, regex=False)
            .str.strip()
        )

    # Build text field from available narrative columns
    cols = [c for c in (key_text_cols or CANDIDATE_TEXT_COLS) if c in df.columns]
    df["text"] = df.apply(lambda r: build_profile_text(r, cols), axis=1).astype(str).str.strip()
    df = df[df["text"].str.len() >= min_chars].copy()

    # Deterministic ID (not shown in UI)
    def _mkid(r):
        basis = f"{r.get('pi_name','')}|{r.get('source_url','')}".strip()
        if basis:
            return hashlib.md5(basis.encode("utf-8")).hexdigest()
        return f"row-{int(r.name)}"

    df["id"] = df.apply(_mkid, axis=1)
    df["name"] = df.get("pi_name", "").astype(str).fillna("")
    df["department"] = df.get("department", "").astype(str).fillna("")

    return df[["id", "name", "email", "department", "text"]].drop_duplicates("id").reset_index(drop=True)


# ------------------------- Embeddings & index -------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (safe, deterministic)."""
    n = np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
    return (v / n).astype(np.float32)

# def build_vector_space(df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
#     """
#     Build a cosine-similarity search index over profile texts.
#     Returns a bundle dict with model, vectors, index, ids, names, texts.
#     """
#     model = SentenceTransformer(model_name)
#     texts = df["text"].tolist()
#     ids   = df["id"].tolist()
#     names = df["name"].tolist()
#     emails = df["emails"].tolist()

#     vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
#     vecs = l2_normalize(np.asarray(vecs))

#     # Keep it minimal & portable: sklearn only
#     index = NearestNeighbors(metric="cosine")
#     index.fit(vecs)

#     return {
#         "model": model,
#         "vectors": vecs,
#         "index": index,
#         "ids": ids,
#         "names": names,
#         "texts": texts
#     }
def build_vector_space(df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    """
    Build a cosine-similarity search index over profile texts.
    Returns a bundle dict with model, vectors, index, ids, names, emails, departments, texts.
    """
    model = SentenceTransformer(model_name)
    texts = df["text"].tolist()
    ids   = df["id"].tolist()
    names = df["name"].tolist()
    emails = df["email"].tolist()
    departments = df["department"].tolist()

    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vecs = l2_normalize(np.asarray(vecs))

    index = NearestNeighbors(metric="cosine")
    index.fit(vecs)

    return {
        "model": model,
        "vectors": vecs,
        "index": index,
        "ids": ids,
        "names": names,
        "emails": emails,
        "departments": departments,
        "texts": texts
    }
def embed_person_text(writeup: str, bundle: Dict) -> np.ndarray:
    """Embed a student's write-up into the same vector space (normalized)."""
    q = bundle["model"].encode([writeup], convert_to_numpy=True, show_progress_bar=False)
    return l2_normalize(q)

# def top_k_matches(query_vec: np.ndarray, bundle: Dict, k: int = 5, min_similarity: float = 0.0) -> pd.DataFrame:
#     """
#     Search the index and return the top-k profiles sorted by cosine similarity.
#     Returns a DataFrame with rank, id, name, similarity, snippet.
#     """
#     ids, names, texts, emails = bundle["ids"], bundle["names"], bundle["texts"], bundle.get("emails", [])
#     k = int(min(max(k, 1), len(ids)))
#     dists, idxs = bundle["index"].kneighbors(query_vec, n_neighbors=k)
#     cosines = (1.0 - dists[0])
#     idxs = idxs[0]
#     rows = []
#     for r, (sim, i) in enumerate(zip(cosines, idxs), start=1):
#         if sim < float(min_similarity):
#             continue
#         snippet = texts[i] if len(texts[i]) <= 500 else (texts[i][:500] + "…")
#         rows.append({"rank": r, "name": names[i], "similarity": float(sim), "snippet": snippet}) #"id": ids[i], "name": names[i], "similarity": float(sim), "snippet": snippet})
#     return pd.DataFrame(rows, columns=["rank","name","similarity","snippet"])#"id","name","similarity","snippet"])
def top_k_matches(query_vec: np.ndarray, bundle: Dict, k: int = 5, min_similarity: float = 0.0) -> pd.DataFrame:
    """
    Search the index and return the top-k profiles sorted by cosine similarity.
    Returns a DataFrame with rank, id, name, department, email, similarity, snippet (≤500 chars).
    """
    ids, names, emails, depts, texts = (
        bundle["ids"], bundle["names"], bundle["emails"], bundle["departments"], bundle["texts"]
    )
    k = int(min(max(k, 1), len(ids)))
    dists, idxs = bundle["index"].kneighbors(query_vec, n_neighbors=k)
    cosines = (1.0 - dists[0])
    idxs = idxs[0]

    rows = []
    for r, (sim, i) in enumerate(zip(cosines, idxs), start=1):
        if sim < float(min_similarity):
            continue
        raw_text = texts[i]
        snippet = raw_text if len(raw_text) <= 500 else (raw_text[:500] + "…")
        rows.append({
            "rank": r,
            "id": ids[i],
            "name": names[i],
            "department": depts[i],
            "email": emails[i],
            "similarity": float(sim),
            "snippet": snippet
        })

    return pd.DataFrame(rows, columns=["rank", "id", "name", "department", "email", "similarity", "snippet"])
def generate_bundle(csv_path: str = "data/Prototype Dataset.csv", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    """Convenience pipeline: load CSV → prep text → build index."""
    df = load_profiles_csv_prototype(csv_path)
    return build_vector_space(df, model_name=model_name)

def score_student_response(writeup: str, bundle: Dict, k: int = 5, min_similarity: float = 0.0) -> pd.DataFrame:
    """Pipeline: embed write-up → retrieve top-k table."""
    qv = embed_person_text(writeup, bundle)
    return top_k_matches(qv, bundle, k=k, min_similarity=min_similarity)
