"""
3GPP Identifier Retriever
-------------------------

Exact retrieval for technical identifiers such as:

    T300
    T301
    T304
    T310
    T311
    RRCReconfiguration
"""

import re


def extract_identifiers(query):
    """
    Extract important 3GPP identifiers.

    Examples
    --------

    What is T300 expiry?
        -> ['t300']

    What happens when T311 expires?
        -> ['t311']

    Compare T300 and T311 expiry.
        -> ['t300', 't311']

    Explain RRCReconfiguration.
        -> ['rrcreconfiguration']
    """

    patterns = [

        # Timers
        r"\bT\d+[A-Za-z]*\b",

        # RRC procedures
        r"\bRRC[A-Za-z]+\b",

        # Technical identifiers such as X-123
        r"\b[A-Za-z]+-\d+\b"
    ]

    identifiers = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            query,
            flags=re.IGNORECASE
        )

        for match in matches:

            value = match.lower()

            if value not in identifiers:

                identifiers.append(
                    value
                )

    return identifiers


def identifier_matches_text(
    text,
    identifier
):
    """
    Exact identifier matching.

    Prevents:

        T300

    from matching:

        T3000
        AT300
        T300abc
    """

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(
            identifier.lower()
        )
        + r"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        str(text).lower(),
        flags=re.IGNORECASE
    ) is not None


def get_matched_identifiers(
    query,
    item
):
    """
    Return requested identifiers
    present in a document.
    """

    identifiers = extract_identifiers(
        query
    )

    metadata = item.get(
        "metadata",
        {}
    )

    title = str(
        metadata.get(
            "title",
            ""
        )
    )

    section = str(
        metadata.get(
            "section",
            ""
        )
    )

    text = str(
        item.get(
            "text",
            ""
        )
    )

    chunk_id = str(
        item.get(
            "chunk_id",
            ""
        )
    )

    searchable_text = (
        title
        + " "
        + section
        + " "
        + text
        + " "
        + chunk_id
    )

    matched = []

    for identifier in identifiers:

        if identifier_matches_text(
            searchable_text,
            identifier
        ):

            matched.append(
                identifier
            )

    return matched


def identifier_search(
    query,
    metadata
):
    """
    Search the complete metadata collection
    for exact requested identifiers.

    Returns
    -------

    dict

    {
        document_index: {
            "identifier_score": 1,
            "matched_identifiers": ["t300"]
        }
    }
    """

    identifiers = extract_identifiers(
        query
    )

    results = {}

    if not identifiers:
        return results

    for idx, item in enumerate(
        metadata
    ):

        matched = get_matched_identifiers(
            query,
            item
        )

        if not matched:
            continue

        results[idx] = {

            "identifier_score":
                float(len(matched)),

            "matched_identifiers":
                matched
        }

    return results