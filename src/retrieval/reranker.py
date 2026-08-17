"""
3GPP RAG - Identifier-aware cross-encoder reranker.

The cross encoder provides semantic relevance, but exact 3GPP
identifiers are treated as a hard ranking signal.

Example:
    Query: "What happens when T311 expires?"
    T311 evidence must outrank T320/T331 evidence.
"""

from sentence_transformers import CrossEncoder
import re


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    print(f"Loading reranker: {MODEL_NAME}")
    return CrossEncoder(MODEL_NAME)


def _extract_timers(query):
    return [
        x.lower()
        for x in re.findall(r"\bT\d+[A-Za-z]*\b", str(query), re.I)
    ]


def _exact_identifier_match(query, candidate):
    identifiers = _extract_timers(query)
    if not identifiers:
        return []

    metadata = candidate.get("metadata", {})
    title = str(metadata.get("title", ""))
    section = str(metadata.get("section", ""))
    text = str(candidate.get("text", ""))
    chunk_id = str(candidate.get("chunk_id", ""))

    searchable = f"{title} {section} {text} {chunk_id}"
    matched = []

    for identifier in identifiers:
        pattern = rf"(?<![a-z0-9]){re.escape(identifier)}(?![a-z0-9])"
        if re.search(pattern, searchable, re.I):
            matched.append(identifier)

    return matched


def rerank(query, candidates, model, top_k=5):
    if not candidates:
        return []

    pairs = []
    for candidate in candidates:
        text = str(candidate.get("text", ""))[:4000]
        pairs.append([query, text])

    print("Running cross-encoder reranking...")
    scores = model.predict(pairs)

    results = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)

        matched = _exact_identifier_match(query, result)
        result["identifiers"] = matched
        result["identifier_exact_match"] = bool(matched)

        # Large deterministic boost prevents semantic similarity
        # from putting another timer above the requested timer.
        identifier_boost = 0.0
        if matched:
            identifier_boost = 20.0 + 5.0 * len(matched)

            title = str(result.get("metadata", {}).get("title", ""))
            if any(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(x)}(?![a-z0-9])",
                    title,
                    re.I,
                )
                for x in matched
            ):
                identifier_boost += 20.0

        result["identifier_boost"] = identifier_boost
        result["rerank_score"] = float(score) + identifier_boost
        results.append(result)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]
