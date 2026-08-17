"""
3GPP Domain-Aware Scorer

Uses custom query extraction to score retrieved chunks.

Signals:

1. Timer match
2. Procedure match
3. Event match
4. Technical identifier match
5. Section relevance
6. Title relevance
"""

import re

from src.retrieval.custom_extractor import (
    extract_query_concepts
)


# ============================================================
# PROCEDURE -> IMPORTANT 3GPP SECTIONS
# ============================================================

PROCEDURE_SECTIONS = {

    "rrc_connection_establishment": [
        "5.3.3",
    ],

    "rrc_connection_reestablishment": [
        "5.3.7",
    ],

    "rrc_connection_resume": [
        "5.3.13",
    ],

    "radio_link_failure": [
        "5.3.10",
    ],

    "security_mode": [
        "5.3.4",
    ],

    "handover_failure": [
        "5.3.5",
    ],
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    return str(text).lower().strip()


# ============================================================
# EXACT TERM MATCH
# ============================================================

def exact_match(text, term):

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(term.lower())
        + r"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        normalize_text(text)
    ) is not None


# ============================================================
# TIMER MATCH
# ============================================================

def timer_score(
    concepts,
    title,
    text,
    chunk_id
):
    """
    Score timer matches.

    Title:
        +20

    Text:
        +8

    Chunk ID:
        +5
    """

    score = 0.0

    searchable_title = normalize_text(title)
    searchable_text = normalize_text(text)
    searchable_chunk = normalize_text(chunk_id)

    for timer in concepts["timers"]:

        timer_lower = timer.lower()

        if exact_match(
            searchable_title,
            timer_lower
        ):
            score += 20.0

        elif exact_match(
            searchable_text,
            timer_lower
        ):
            score += 8.0

        elif exact_match(
            searchable_chunk,
            timer_lower
        ):
            score += 5.0

    return score


# ============================================================
# PROCEDURE MATCH
# ============================================================

def procedure_score(
    concepts,
    section,
    title,
    text
):
    """
    Score domain-specific procedures.

    Exact procedure in title:
        +25

    Procedure in text:
        +12

    Correct 3GPP section:
        +20
    """

    score = 0.0

    section = normalize_text(section)
    title = normalize_text(title)
    text = normalize_text(text)

    for procedure in concepts["procedures"]:

        readable = procedure.replace(
            "_",
            " "
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        if readable in title:

            score += 25.0

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        elif readable in text:

            score += 12.0

        # ----------------------------------------------------
        # SECTION
        # ----------------------------------------------------

        important_sections = (
            PROCEDURE_SECTIONS.get(
                procedure,
                []
            )
        )

        for important_section in important_sections:

            if (
                section == important_section
                or section.startswith(
                    important_section + "."
                )
            ):

                score += 20.0
                break

    return score


# ============================================================
# EVENT SCORE
# ============================================================

def event_score(
    concepts,
    title,
    text
):
    """
    Score events such as:

        expiry
        failure
        timeout
        rejection
    """

    score = 0.0

    title = normalize_text(title)
    text = normalize_text(text)

    for event in concepts["events"]:

        if exact_match(
            title,
            event
        ):
            score += 8.0

        elif exact_match(
            text,
            event
        ):
            score += 3.0

    return score


# ============================================================
# TECHNICAL IDENTIFIER SCORE
# ============================================================

def technical_identifier_score(
    concepts,
    title,
    text,
    chunk_id
):
    """
    Score technical identifiers such as:

        RRCSetup
        RRCReconfiguration
        connEstFailCount
    """

    score = 0.0

    for identifier in concepts[
        "technical_identifiers"
    ]:

        if exact_match(
            title,
            identifier
        ):
            score += 15.0

        elif exact_match(
            text,
            identifier
        ):
            score += 6.0

        elif exact_match(
            chunk_id,
            identifier
        ):
            score += 4.0

    return score


# ============================================================
# QUERY/TITLE OVERLAP
# ============================================================

def title_overlap(
    query,
    title
):
    """
    Small general semantic signal.

    Prevents domain-specific scoring from
    completely ignoring normal query words.
    """

    query_words = set(
        re.findall(
            r"[a-z0-9]+",
            query.lower()
        )
    )

    title_words = set(
        re.findall(
            r"[a-z0-9]+",
            title.lower()
        )
    )

    if not query_words:
        return 0.0

    overlap = (
        len(query_words & title_words)
        / len(query_words)
    )

    return overlap * 5.0


# ============================================================
# MAIN DOMAIN SCORE
# ============================================================

def calculate_domain_score(
    query,
    item
):
    """
    Calculate complete 3GPP-aware score.

    Returns:

        total score
    """

    concepts = extract_query_concepts(
        query
    )

    metadata = item.get(
        "metadata",
        {}
    )

    section = str(
        metadata.get(
            "section",
            ""
        )
    )

    title = str(
        metadata.get(
            "title",
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

    # ========================================================
    # COMPONENT SCORES
    # ========================================================

    timer = timer_score(
        concepts,
        title,
        text,
        chunk_id
    )

    procedure = procedure_score(
        concepts,
        section,
        title,
        text
    )

    event = event_score(
        concepts,
        title,
        text
    )

    technical = technical_identifier_score(
        concepts,
        title,
        text,
        chunk_id
    )

    overlap = title_overlap(
        query,
        title
    )

    # ========================================================
    # TOTAL
    # ========================================================

    total = (
        timer
        + procedure
        + event
        + technical
        + overlap
    )

    return total


# ============================================================
# DEBUG INFORMATION
# ============================================================

def explain_domain_score(
    query,
    item
):
    """
    Useful during development/interview demo.

    Shows why a document received
    its domain score.
    """

    concepts = extract_query_concepts(
        query
    )

    metadata = item.get(
        "metadata",
        {}
    )

    section = str(
        metadata.get(
            "section",
            ""
        )
    )

    title = str(
        metadata.get(
            "title",
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

    timer = timer_score(
        concepts,
        title,
        text,
        chunk_id
    )

    procedure = procedure_score(
        concepts,
        section,
        title,
        text
    )

    event = event_score(
        concepts,
        title,
        text
    )

    technical = technical_identifier_score(
        concepts,
        title,
        text,
        chunk_id
    )

    overlap = title_overlap(
        query,
        title
    )

    total = (
        timer
        + procedure
        + event
        + technical
        + overlap
    )

    return {
        "timer_score": timer,
        "procedure_score": procedure,
        "event_score": event,
        "technical_identifier_score": technical,
        "title_overlap_score": overlap,
        "total_score": total,
    }