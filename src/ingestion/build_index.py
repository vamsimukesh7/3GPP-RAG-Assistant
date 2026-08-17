"""
3GPP RAG - FAISS Vector Index Builder 

Input:
    data/chunks.json

Output:
    vectorstore/
        index.faiss
        metadata.json
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/chunks.json")
VECTORSTORE_DIR = Path("vectorstore")

INDEX_FILE = VECTORSTORE_DIR / "index.faiss"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"

# Good multilingual/general-purpose embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading chunks.json...")

    data = json.loads(
        INPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    chunks = data["chunks"]

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    print()
    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print()
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # Because embeddings are normalized,
    # Inner Product is equivalent to cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    print("Adding vectors to FAISS...")

    index.add(
        embeddings
    )

    print(
        f"Vectors in index: {index.ntotal}"
    )

    print("Saving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Store metadata separately.
    metadata = []

    for chunk in chunks:

        metadata.append({

            "chunk_id":
                chunk["chunk_id"],

            "text":
                chunk["text"],

            "metadata":
                chunk["metadata"]
        })

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("========== SUCCESS ==========")

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Dimension: {dimension}"
    )

    print(
        f"FAISS index: {INDEX_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()