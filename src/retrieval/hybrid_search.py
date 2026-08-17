"""
3GPP RAG - Final Hybrid Retrieval

Pipeline:
Query
 -> requested/reference identifier extraction
 -> FAISS dense retrieval
 -> BM25 retrieval
 -> exact identifier retrieval
 -> domain scoring
 -> identifier-aware cross encoder
 -> final score
 -> hallucination/evidence guard

Important:
- Requested identifiers are mandatory evidence.
- Reference identifiers help retrieval but are not mandatory.
- Missing T999 must remain UNSUPPORTED even when T320/T331 are
  semantically similar.
"""

import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.reranker import load_reranker, rerank
from src.retrieval.domain_scorer import calculate_domain_score
from src.guardrails.hallucination_guard import (
    validate_evidence,
    print_validation_result,
)

INDEX_FILE = PROJECT_ROOT / "vectorstore" / "index.faiss"
METADATA_FILE = PROJECT_ROOT / "vectorstore" / "metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DENSE_TOP_K = 10
BM25_TOP_K = 10
HYBRID_TOP_K = 15
FINAL_TOP_K = 5


def tokenize(text):
    return re.findall(
        r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*",
        str(text).lower(),
    )


def extract_all_identifiers(query):
    patterns = [
        r"\bT\d+[A-Za-z]*\b",
        r"\bRRC[A-Za-z]+\b",
        r"\b[A-Za-z]+-\d+\b",
    ]

    identifiers = []
    for pattern in patterns:
        for match in re.findall(pattern, str(query), re.I):
            value = match.lower().strip()
            if value not in identifiers:
                identifiers.append(value)

    return identifiers


def extract_requested_identifiers(query):
    all_ids = extract_all_identifiers(query)
    if not all_ids:
        return []

    q = str(query).lower()

    reference_patterns = [
        r"\buse\s+((?:t\d+[a-z]*))",
        r"\busing\s+((?:t\d+[a-z]*))",
        r"\brefer\s+to\s+((?:t\d+[a-z]*))",
        r"\breference\s+((?:t\d+[a-z]*))",
        r"\bbased\s+on\s+((?:t\d+[a-z]*))",
        r"\bcompare\s+with\s+((?:t\d+[a-z]*))",
        r"\buse\s+((?:t\d+[a-z]*))\s+as\s+reference",
    ]

    references = set()
    for pattern in reference_patterns:
        for match in re.findall(pattern, q, re.I):
            references.add(match.lower())

    requested = [x for x in all_ids if x not in references]

    if not requested:
        expiry_matches = re.findall(
            r"\b(T\d+[A-Za-z]*)\b(?=.*\bexpir)",
            q,
            re.I,
        )
        for identifier in expiry_matches:
            identifier = identifier.lower()
            if identifier not in requested:
                requested.append(identifier)

    return requested


def extract_reference_identifiers(query):
    all_ids = set(extract_all_identifiers(query))
    requested = set(extract_requested_identifiers(query))
    return sorted(all_ids - requested)


def extract_identifiers(query):
    """Backward-compatible alias: only mandatory/requested identifiers."""
    return extract_requested_identifiers(query)


def identifier_matches_text(text, identifier):
    pattern = rf"(?<![a-z0-9]){re.escape(identifier.lower())}(?![a-z0-9])"
    return re.search(pattern, str(text).lower(), re.I) is not None


def get_matched_identifiers(query, item, include_references=True):
    identifiers = extract_requested_identifiers(query)

    if include_references:
        identifiers += [
            x for x in extract_reference_identifiers(query)
            if x not in identifiers
        ]

    metadata = item.get("metadata", {})
    searchable = " ".join([
        str(metadata.get("title", "")),
        str(metadata.get("section", "")),
        str(item.get("text", "")),
        str(item.get("chunk_id", "")),
    ])

    return [
        identifier
        for identifier in identifiers
        if identifier_matches_text(searchable, identifier)
    ]


def load_resources():
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))
    print(f"FAISS vectors: {index.ntotal}")

    print("Loading metadata...")
    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )
    print(f"Metadata records: {len(metadata)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building BM25 index...")
    corpus = []
    for item in metadata:
        m = item.get("metadata", {})
        combined = " ".join([
            str(m.get("section", "")),
            str(m.get("title", "")),
            str(item.get("text", "")),
        ])
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    print(f"BM25 documents: {len(corpus)}")

    print("Loading cross-encoder reranker...")
    reranker_model = load_reranker()

    return index, metadata, embedding_model, bm25, reranker_model


def dense_search(query, index, embedding_model, top_k=DENSE_TOP_K):
    vector = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )
    vector = np.asarray(vector, dtype=np.float32)

    scores, indices = index.search(
        vector,
        min(top_k, index.ntotal),
    )

    results = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results[int(idx)] = {"dense_score": float(score)}
    return results


def bm25_search(query, bm25, top_k=BM25_TOP_K):
    scores = bm25.get_scores(tokenize(query))
    indices = np.argsort(scores)[::-1][:top_k]

    return {
        int(idx): {"bm25_score": float(scores[idx])}
        for idx in indices
    }


def normalize_scores(results, score_key):
    values = [
        x[score_key]
        for x in results.values()
        if score_key in x
    ]

    if not values:
        return

    lo, hi = min(values), max(values)

    if hi == lo:
        for item in results.values():
            if score_key in item:
                item[score_key + "_norm"] = 1.0
        return

    for item in results.values():
        if score_key in item:
            item[score_key + "_norm"] = (
                item[score_key] - lo
            ) / (hi - lo)


def hybrid_search(
    query,
    index,
    metadata,
    embedding_model,
    bm25,
    top_k=HYBRID_TOP_K,
):
    requested = extract_requested_identifiers(query)
    references = extract_reference_identifiers(query)

    dense = dense_search(query, index, embedding_model)
    keyword = bm25_search(query, bm25)

    combined = {}

    for idx, data in dense.items():
        combined.setdefault(idx, {}).update(data)

    for idx, data in keyword.items():
        combined.setdefault(idx, {}).update(data)

    # Exact identifier retrieval.
    # Search the whole metadata set so a true T311 chunk can enter
    # the pool even if FAISS/BM25 missed it.
    search_ids = requested + [
        x for x in references if x not in requested
    ]

    for idx, item in enumerate(metadata):
        matched = get_matched_identifiers(
            query,
            item,
            include_references=True,
        )

        if matched:
            combined.setdefault(idx, {})
            combined[idx]["matched_identifiers"] = matched
            combined[idx]["identifier_score"] = float(len(matched))

    normalize_scores(combined, "dense_score")
    normalize_scores(combined, "bm25_score")

    for idx, scores in combined.items():
        item = metadata[idx]
        matched = scores.get("matched_identifiers", [])

        domain_score = calculate_domain_score(
            query,
            item,
        )

        # Exact requested identifier match is stronger than
        # semantic similarity. Reference identifiers receive
        # only a smaller contextual boost.
        requested_hits = [
            x for x in matched
            if x in requested
        ]
        reference_hits = [
            x for x in matched
            if x in references
        ]

        exact_score = (
            60.0 * len(requested_hits)
            + 10.0 * len(reference_hits)
        )

        title = str(item.get("metadata", {}).get("title", ""))
        title_hits = sum(
            1
            for x in requested
            if identifier_matches_text(title, x)
        )

        exact_score += 40.0 * title_hits

        dense_norm = scores.get("dense_score_norm", 0.0)
        bm25_norm = scores.get("bm25_score_norm", 0.0)

        hybrid_score = (
            0.25 * dense_norm
            + 0.20 * bm25_norm
            + 0.55 * min(exact_score / 100.0, 1.0)
        )

        scores["domain_score"] = float(domain_score)
        scores["identifier_score"] = float(exact_score)
        scores["hybrid_score"] = float(hybrid_score)
        scores["identifier_coverage"] = (
            len(requested_hits) / len(requested)
            if requested else 1.0
        )
        scores["matched_identifiers"] = matched

    ranked = sorted(
        combined.items(),
        key=lambda pair: (
            pair[1].get("identifier_score", 0.0),
            pair[1].get("hybrid_score", 0.0),
            pair[1].get("domain_score", 0.0),
        ),
        reverse=True,
    )

    candidates = []

    for idx, scores in ranked[:top_k]:
        item = metadata[idx]
        matched = scores.get("matched_identifiers", [])

        candidates.append({
            "index": idx,
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text", ""),
            "metadata": item.get("metadata", {}),
            "identifiers": [
                x for x in matched if x in requested
            ],
            "scores": scores,
        })

    # Calculate coverage from exact requested evidence.
    covered = set()
    for candidate in candidates:
        covered.update(candidate.get("identifiers", []))

    missing = [
        x for x in requested
        if x not in covered
    ]

    coverage = (
        len(covered) / len(requested)
        if requested else 1.0
    )

    for candidate in candidates:
        candidate["query_identifier_coverage"] = coverage
        candidate["covered_identifiers"] = sorted(covered)
        candidate["missing_identifiers"] = missing
        candidate["requested_identifiers"] = requested
        candidate["reference_identifiers"] = references

    return candidates



def expand_primary_section(ranked_results, metadata, max_section_chunks=8):
    """Expand the top-ranked 3GPP section to include sibling chunks."""
    if not ranked_results:
        return ranked_results
    primary = ranked_results[0]
    section = primary.get("metadata", {}).get("section")
    if not section:
        return ranked_results
    originals = {r.get("chunk_id"): r for r in ranked_results}
    expanded = []
    requested = primary.get("requested_identifiers", [])
    references = primary.get("reference_identifiers", [])
    for idx, item in enumerate(metadata):
        if item.get("metadata", {}).get("section") != section:
            continue
        cid = item.get("chunk_id")
        if cid in originals:
            r = dict(originals[cid])
            r["section_expanded"] = True
        else:
            r = {
                "index": idx, "chunk_id": cid, "text": item.get("text", ""),
                "metadata": item.get("metadata", {}), "scores": {},
                "reranker_score": 0.0, "final_score": primary.get("final_score", 0.0) * 0.90,
                "authority_score": primary.get("authority_score", 0.0),
                "domain_score": primary.get("domain_score", 0.0),
                "hybrid_score": primary.get("hybrid_score", 0.0),
                "identifier_coverage": primary.get("identifier_coverage", 1.0),
                "requested_identifiers": requested, "reference_identifiers": references,
                "section_expanded": True,
            }
        searchable = " ".join([str(item.get("metadata", {}).get("title", "")), str(item.get("metadata", {}).get("section", "")), str(item.get("text", "")), str(cid or "")])
        r["identifiers"] = [x for x in requested + [y for y in references if y not in requested] if identifier_matches_text(searchable, x)]
        expanded.append(r)
    def order(r):
        try: return int(r.get("metadata", {}).get("chunk_index", 0))
        except (TypeError, ValueError): return 0
    expanded.sort(key=order)
    return expanded[:max_section_chunks]

def run_query(
    query,
    index,
    metadata,
    embedding_model,
    bm25,
    reranker_model,
):
    candidates = hybrid_search(
        query,
        index,
        metadata,
        embedding_model,
        bm25,
        HYBRID_TOP_K,
    )

    requested = extract_requested_identifiers(query)
    references = extract_reference_identifiers(query)

    print(f"\nHybrid candidates: {len(candidates)}")
    print(f"Requested identifiers: {requested}")
    print(f"Reference identifiers: {references}")

    if candidates:
        print(
            f"Query identifier coverage: "
            f"{candidates[0]['query_identifier_coverage']:.2f}"
        )
        print(f"Covered: {candidates[0]['covered_identifiers']}")
        print(f"Missing: {candidates[0]['missing_identifiers']}")

    results = rerank(
        query,
        candidates,
        reranker_model,
        FINAL_TOP_K,
    )

    for result in results:
        result["query"] = query

    for result in results:
        scores = result.get("scores", {})
        reranker_score = result.get("rerank_score", 0.0)

        result["authority_score"] = float(
            scores.get("authority_score", 0.0)
        )
        result["domain_score"] = float(
            scores.get("domain_score", 0.0)
        )
        result["hybrid_score"] = float(
            scores.get("hybrid_score", 0.0)
        )
        result["identifier_coverage"] = float(
            scores.get("identifier_coverage", 0.0)
        )

        # Identifier-aware final score.
        # Exact identifier evidence gets deterministic priority.
        result["final_score"] = (
            0.50 * reranker_score
            + 0.20 * result["domain_score"]
            + 0.15 * result["authority_score"]
            + 15.0 * result["identifier_coverage"]
        )

        result["requested_identifiers"] = requested
        result["reference_identifiers"] = references

    results.sort(
        key=lambda x: x.get("final_score", 0.0),
        reverse=True,
    )

    # Keep coverage metadata after reranking.
    for result in results:
        if candidates:
            result["query_identifier_coverage"] = candidates[0][
                "query_identifier_coverage"
            ]
            result["covered_identifiers"] = candidates[0][
                "covered_identifiers"
            ]
            result["missing_identifiers"] = candidates[0][
                "missing_identifiers"
            ]

    # Expand the highest-ranked section before evidence validation.
    # This keeps multi-chunk sections together (e.g. _001, _002).
    results = expand_primary_section(
        results,
        metadata,
        max_section_chunks=8,
    )

    print("\nPrimary section expansion:")
    if results:
        primary_section = results[0].get("metadata", {}).get("section")
        print(f"Primary section: {primary_section}")
        print("Section chunks:")
        for result in results:
            print(f"  {result.get('chunk_id')}")

    # Recompute exact identifier coverage on the expanded evidence.
    covered = set()
    for result in results:
        covered.update(x for x in result.get("identifiers", []) if x in requested)
    missing = [x for x in requested if x not in covered]
    coverage = len(covered) / len(requested) if requested else 1.0
    for result in results:
        result["query_identifier_coverage"] = coverage
        result["covered_identifiers"] = sorted(covered)
        result["missing_identifiers"] = missing
        result["requested_identifiers"] = requested
        result["reference_identifiers"] = references

    guard_result = validate_evidence(
        query=query,
        results=results,
        requested_identifiers=requested,
    )

    return results, guard_result


def display_results(query, results, guard_result):
    print("\n" + "=" * 80)
    print("FINAL 3GPP-AWARE RESULTS")
    print("=" * 80)

    print(f"Query: {query}")
    print(f"Requested identifiers: {guard_result.get('identifiers', [])}")
    print(f"Covered identifiers: {guard_result.get('covered_identifiers', [])}")
    print(f"Missing identifiers: {guard_result.get('missing_identifiers', [])}")

    print_validation_result(guard_result)

    for rank, result in enumerate(results, 1):
        metadata = result.get("metadata", {})

        print(f"\n[{rank}] Final Score: {result.get('final_score', 0):.4f}")
        print(f"Reranker Score: {result.get('reranker_score', 0):.4f}")
        print(f"Identifier Boost: {result.get('identifier_boost', 0):.4f}")
        print(f"Section: {metadata.get('section')}")
        print(f"Title: {metadata.get('title')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Matched identifiers: {result.get('identifiers', [])}")
        print("\n" + str(result.get("text", ""))[:1500])
        print("-" * 80)


def main():
    resources = load_resources()

    print("\n" + "=" * 80)
    print("3GPP HYBRID + RERANKER + GUARDRAIL TESTER")
    print("=" * 80)
    print("Type 'exit' to stop.")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results, guard_result = run_query(
            query,
            *resources,
        )

        display_results(
            query,
            results,
            guard_result,
        )


if __name__ == "__main__":
    main()