import os
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import cohere
    COHERE_AVAILABLE = bool(os.getenv("COHERE_API_KEY"))
except ImportError:
    COHERE_AVAILABLE = False

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False


def _vector_search(query_vec: np.ndarray, kb: dict, top_k: int) -> list[dict]:
    """Run FAISS vector search, return top_k chunk dicts."""
    if kb["index"] is None or not kb["chunks"]:
        return []
    distances, indices = kb["index"].search(
        query_vec.astype("float32"), min(top_k, len(kb["chunks"]))
    )
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(kb["chunks"]):
            results.append(kb["chunks"][idx])
    return results


def _bm25_search(query: str, kb: dict, top_k: int) -> list[dict]:
    """BM25 keyword search. Only available for user KB."""
    if not kb.get("bm25") or not kb["chunks"]:
        return []
    tokenized_query = query.lower().split()
    scores = kb["bm25"].get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [kb["chunks"][i] for i in top_indices if scores[i] > 0]


def _rerank_cohere(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank using Cohere API."""
    import cohere
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    texts = [c["text"] for c in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_k,
    )
    reranked = []
    for r in response.results:
        chunk = candidates[r.index].copy()
        chunk["score"] = r.relevance_score
        reranked.append(chunk)
    return reranked


def _rerank_flashrank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank using local FlashRank (no API key needed)."""
    from flashrank import Ranker, RerankRequest
    ranker = Ranker()
    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    reranked = []
    for r in results[:top_k]:
        chunk = candidates[r["id"]].copy()
        chunk["score"] = r.get("score", 0)
        reranked.append(chunk)
    return reranked


def _rerank_simple(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Fallback: simple keyword overlap scoring."""
    query_words = set(query.lower().split())
    scored = []
    for c in candidates:
        text_words = set(c["text"].lower().split())
        score = len(query_words & text_words) / (len(query_words) + 1)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def hybrid_retrieve(
    query: str,
    user_kb: dict,
    med_kb: dict,
    embed_model: SentenceTransformer,
    top_k_final: int = 5,
    top_k_candidates: int = 15,
) -> dict:
    """
    Full hybrid retrieval pipeline:
    1. Vector search on both KBs
    2. BM25 on user KB
    3. Merge 50/50 (up to top_k_candidates total)
    4. Rerank → return top_k_final split between sources

    Returns:
        {
            "pdf_chunks": [list of top PDF chunk dicts],
            "med_chunks": [list of top MED chunk dicts],
        }
    """
    query_vec = embed_model.encode([query])

    # Vector search
    user_vec_results = _vector_search(query_vec, user_kb, top_k_candidates // 2)
    med_vec_results = _vector_search(query_vec, med_kb, top_k_candidates // 2)

    # BM25 on user KB only (medical KB has no BM25 by design - too large)
    user_bm25_results = _bm25_search(query, user_kb, top_k_candidates // 4)

    # Merge user candidates (deduplicate by text)
    seen_texts = set()
    user_candidates = []
    for chunk in user_vec_results + user_bm25_results:
        if chunk["text"] not in seen_texts:
            seen_texts.add(chunk["text"])
            user_candidates.append(chunk)

    med_candidates = med_vec_results

    # Rerank each pool separately
    all_candidates = user_candidates + med_candidates

    if not all_candidates:
        return {"pdf_chunks": [], "med_chunks": []}

    # Choose reranker
    try:
        if COHERE_AVAILABLE:
            reranked = _rerank_cohere(query, all_candidates, top_k_final * 2)
        elif FLASHRANK_AVAILABLE:
            reranked = _rerank_flashrank(query, all_candidates, top_k_final * 2)
        else:
            reranked = _rerank_simple(query, all_candidates, top_k_final * 2)
    except Exception as e:
        print(f"[Retrieval] Reranker failed ({e}), falling back to simple scoring")
        reranked = _rerank_simple(query, all_candidates, top_k_final * 2)

    # Split by source tag
    pdf_chunks = [c for c in reranked if c.get("tag") == "PDF"][:3]
    med_chunks = [c for c in reranked if c.get("tag") == "MED"][:3]

    return {"pdf_chunks": pdf_chunks, "med_chunks": med_chunks}