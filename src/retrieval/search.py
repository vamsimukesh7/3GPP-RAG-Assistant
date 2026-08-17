"""
3GPP RAG - FAISS Retrieval Tester

Usage:
    python src/retrieval/search.py

Then enter questions interactively.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("vectorstore/index.faiss")
METADATA_FILE = Path("vectorstore/metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


def load_resources():

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"Vectors loaded: {index.ntotal}"
    )

    print("Loading metadata...")

    metadata = json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"Metadata records: {len(metadata)}"
    )

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    return index, metadata, model


def search(
    query,
    index,
    metadata,
    model,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = metadata[idx]

        results.append({

            "score":
                float(score),

            "chunk_id":
                item["chunk_id"],

            "text":
                item["text"],

            "metadata":
                item["metadata"]
        })

    return results


def display_results(
    query,
    results
):

    print()
    print("=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    print()
    print("=" * 80)
    print("RETRIEVED EVIDENCE")
    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print()
        print(
            f"[{i}] "
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Section: "
            f"{metadata['section']}"
        )

        print(
            f"Title: "
            f"{metadata['title']}"
        )

        print(
            f"Page: "
            f"{metadata['page']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print()
        print(
            result["text"][:1500]
        )

        print()
        print("-" * 80)


def main():

    index, metadata, model = (
        load_resources()
    )

    print()
    print("=" * 80)
    print("3GPP RAG RETRIEVAL TESTER")
    print("=" * 80)

    print(
        "Enter a question."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        print()

        query = input(
            "Question: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit"
        }:

            break

        if not query:
            continue

        results = search(
            query,
            index,
            metadata,
            model
        )

        display_results(
            query,
            results
        )


if __name__ == "__main__":
    main()