"""
BM25 Retriever
--------------
Keyword-based retrieval for technical 3GPP text.
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    """
    Tokenize technical 3GPP text.

    Examples:

        T300
        T311
        RRCReconfiguration
        connEstFailCount
        Qoffsettemp
    """

    return re.findall(
        r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*",
        str(text).lower()
    )


def build_bm25(metadata):
    """
    Build BM25 index from document metadata.
    """

    corpus = []

    for item in metadata:

        item_metadata = item.get(
            "metadata",
            {}
        )

        section = str(
            item_metadata.get(
                "section",
                ""
            )
        )

        title = str(
            item_metadata.get(
                "title",
                ""
            )
        )

        text = item.get(
            "text",
            ""
        )

        combined = (
            section
            + " "
            + title
            + " "
            + text
        )

        corpus.append(
            tokenize(combined)
        )

    return BM25Okapi(corpus)


def bm25_search(
    query,
    bm25,
    top_k=10
):
    """
    Perform BM25 keyword retrieval.
    """

    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = {}

    for idx in indices:

        results[int(idx)] = {
            "bm25_score": float(
                scores[idx]
            )
        }

    return results