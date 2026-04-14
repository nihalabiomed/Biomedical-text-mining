"""
semantic_search.py
==================
Bio-NLP Semantic Search — core search function.

Contains the complete semantic_search() pipeline:
  - Load pre-computed embeddings from embeddings.pt or embeddings.pkl
  - Encode a user query (sentence-transformers BioBERT, or TF-IDF fallback)
  - Cosine similarity search over all abstract embeddings
  - Return ranked results with NER entity metadata

Usage (standalone test):
    python semantic_search.py

Usage (import):
    from semantic_search import semantic_search, print_results
    results = semantic_search("BRCA1 DNA repair breast cancer", top_k=5)
"""

import os, pickle
import numpy as np
import torch

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME         = "dmis-lab/biobert-base-cased-v1.2"
EMBEDDINGS_PT      = "embeddings.pt"
EMBEDDINGS_PKL     = "embeddings.pkl"
DEFAULT_TOP_K      = 10
VALID_ENTITY_TYPES = {"Disease", "Gene", "Cancer"}

_cache = {}   # module-level cache so embeddings load only once per session


# ── 1. Load pre-computed embeddings ───────────────────────────────────────────

def load_embeddings(path_pt: str = EMBEDDINGS_PT,
                    path_pkl: str = EMBEDDINGS_PKL) -> dict:
    """
    Load abstract embeddings from disk (cached after first call).

    Returns dict with keys:
        embeddings   — (N, 768) float32 numpy array, L2-normalised
        questions    — list[str]
        abstract_ids — list[str]
        entities     — list[list[dict]]
        vectorizer   — fitted TfidfVectorizer (for consistent query encoding)
        embed_dim    — 768
        num_abstracts — N
    """
    if "data" in _cache:
        return _cache["data"]

    if os.path.exists(path_pt):
        print(f"[load] {path_pt}")
        data = torch.load(path_pt, map_location="cpu", weights_only=False)
        if isinstance(data["embeddings"], torch.Tensor):
            data["embeddings"] = data["embeddings"].numpy().astype(np.float32)
    elif os.path.exists(path_pkl):
        print(f"[load] {path_pkl}")
        with open(path_pkl, "rb") as f:
            data = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"No embedding file found. Expected '{path_pt}' or '{path_pkl}'.\n"
            "Run: python generate_embeddings.py"
        )

    _cache["data"] = data
    print(f"[load] {data['num_abstracts']} abstracts, dim={data['embed_dim']}")
    return data


# ── 2. Encode a query ──────────────────────────────────────────────────────────

def encode_query(query: str,
                 data: dict,
                 model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Encode a free-text query into a 768-dim L2-normalised vector.

    Tries sentence-transformers (BioBERT) first; falls back to the
    TF-IDF vectorizer that was fitted when the index was built.

    Returns: (1, 768) float32 numpy array
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import normalize
        print(f"[encode] sentence-transformers: {model_name}")
        model     = SentenceTransformer(model_name)
        query_vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return query_vec                      # (1, 768)

    except Exception:
        # Use the same TF-IDF vectorizer that built the index — guarantees
        # identical feature space and thus meaningful cosine similarity.
        from sklearn.preprocessing import normalize
        vectorizer = data.get("vectorizer")
        if vectorizer is None:
            raise RuntimeError(
                "sentence-transformers unavailable and no vectorizer found in "
                "embeddings file. Rebuild embeddings with generate_embeddings.py."
            )
        query_vec = vectorizer.transform([query]).toarray().astype(np.float32)
        query_vec = normalize(query_vec, norm="l2")
        return query_vec                      # (1, 768)


# ── 3. Cosine similarity search ────────────────────────────────────────────────

def cosine_similarity_search(query_vec: np.ndarray,
                              abstract_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between the query and all abstract vectors.

    Both vectors are L2-normalised, so cosine_sim = dot_product.

    Args:
        query_vec           : (1, D) float32
        abstract_embeddings : (N, D) float32

    Returns:
        scores : (N,) float32  — similarity scores in [-1, 1]
    """
    return (abstract_embeddings @ query_vec.T).squeeze()   # (N,)


# ── 4. Full semantic search pipeline ──────────────────────────────────────────

def semantic_search(
    query: str,
    top_k: int        = DEFAULT_TOP_K,
    entity_type: str  = None,
    path_pt: str      = EMBEDDINGS_PT,
    path_pkl: str     = EMBEDDINGS_PKL,
) -> list:
    """
    Semantic search over all biomedical abstracts.

    Parameters
    ----------
    query       : free-text biomedical query
    top_k       : number of results to return
    entity_type : optional filter — 'Disease', 'Gene', or 'Cancer'
    path_pt     : path to embeddings.pt
    path_pkl    : path to embeddings.pkl (fallback)

    Returns
    -------
    list of dicts, sorted by cosine similarity (descending):
        rank          : int    — 1-indexed result rank
        abstract_id   : str    — source abstract ID
        question      : str    — original biomedical question
        score         : float  — cosine similarity score [-1, 1]
        score_pct     : float  — score as percentage [0, 100]
        entities      : list   — deduplicated entity dicts {entity, type, umls_id}
        entity_counts : dict   — {Disease: int, Gene: int, Cancer: int}
    """
    if entity_type and entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(
            f"entity_type must be one of {sorted(VALID_ENTITY_TYPES)}, got '{entity_type}'"
        )

    # Load index (cached after first call)
    data        = load_embeddings(path_pt, path_pkl)
    embeddings  = data["embeddings"]       # (N, 768)
    questions   = data["questions"]
    abs_ids     = data["abstract_ids"]
    all_entities = data["entities"]

    # Encode query → (1, 768)
    query_vec = encode_query(query, data)

    # Cosine similarity → (N,)
    scores = cosine_similarity_search(query_vec, embeddings)

    # Sort descending
    ranked = np.argsort(scores)[::-1]

    # Collect top_k results (with optional entity filter)
    results = []
    for idx in ranked:
        if len(results) >= top_k:
            break

        entities = all_entities[idx]

        if entity_type:
            matched = [e for e in entities if e["type"] == entity_type]
            if not matched:
                continue
        else:
            matched = entities

        # Deduplicate by (entity text, type)
        seen, dedup = set(), []
        for e in matched:
            key = (e["entity"].lower(), e["type"])
            if key not in seen:
                seen.add(key)
                dedup.append(e)

        results.append({
            "rank":          len(results) + 1,
            "abstract_id":   abs_ids[idx],
            "question":      questions[idx],
            "score":         round(float(scores[idx]), 4),
            "score_pct":     round(float(scores[idx]) * 100, 2),
            "entities":      dedup,
            "entity_counts": {
                "Disease": sum(1 for e in entities if e["type"] == "Disease"),
                "Gene":    sum(1 for e in entities if e["type"] == "Gene"),
                "Cancer":  sum(1 for e in entities if e["type"] == "Cancer"),
            },
        })

    return results


# ── 5. Pretty printer ──────────────────────────────────────────────────────────

def print_results(results: list, query: str):
    """Print results in readable format."""
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  Query : \"{query}\"")
    print(f"  Found : {len(results)} results")
    print(bar)
    for r in results:
        ec = r["entity_counts"]
        tags = " | ".join(
            f"{v} {k}" for k, v in ec.items() if v > 0
        )
        top_ents = ", ".join(
            f"{e['type']}:{e['entity']}" for e in r["entities"][:3]
        )
        print(f"\n  [{r['rank']}] {r['score_pct']:.1f}%  (Abstract #{r['abstract_id']})")
        print(f"       {r['question'][:75]}")
        print(f"       Entities : {tags}")
        print(f"       Top tags : {top_ents}")
    print(f"\n{bar}\n")


# ── Interactive demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_queries = [
        ("EGFR signaling receptor ligands cancer",   None),
        ("Hirschsprung disease genetics mendelian",  "Disease"),
        ("BRCA1 BRCA2 DNA repair homologous",        "Gene"),
        ("breast cancer chemotherapy resistance",    "Cancer"),
        ("insulin type 2 diabetes glucose",          None),
        ("Alzheimer amyloid plaques APOE",           None),
        ("platelet therapy aspirin COX inhibitor",   "Gene"),
    ]

    for query, etype in demo_queries:
        label = f"  [filter: {etype}]" if etype else ""
        print(f"\n>>> Searching: '{query}'{label}")
        results = semantic_search(query, top_k=3, entity_type=etype)
        print_results(results, query + (f" [{etype}]" if etype else ""))
        print("-" * 68)
