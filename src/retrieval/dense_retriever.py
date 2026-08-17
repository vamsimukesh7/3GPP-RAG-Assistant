"""
Dense Retriever
---------------
FAISS-based semantic retrieval.
"""

import numpy as np


def dense_search(
    query,
    index,
    embedding_model,
    top_k=10
):
    """
    Perform semantic retrieval using FAISS.

    Parameters
    ----------
    query : str
        User question.

    index : FAISS index
        Loaded FAISS vector index.

    embedding_model : SentenceTransformer
        Embedding model.

    top_k : int
        Number of results to retrieve.

    Returns
    -------
    dict
        {
            index: {
                "dense_score": float
            }
        }
    """

    query_embedding = embedding_model.encode(
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

    results = {}

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results[int(idx)] = {
            "dense_score": float(score)
        }

    return results