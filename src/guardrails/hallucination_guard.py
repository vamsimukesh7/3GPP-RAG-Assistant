"""
3GPP-aware hallucination/evidence guard.

Rules:
1. If a query explicitly asks for an identifier, that identifier
   MUST be present in the retrieved evidence.
2. Semantic similarity can never replace missing identifier evidence.
3. Reference identifiers are not mandatory.
4. Missing evidence => UNSUPPORTED.
"""

import re


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_section(result):
    metadata = result.get("metadata", {})
    return (
        result.get("section")
        or result.get("section_id")
        or metadata.get("section")
    )


def _extract_identifiers_from_text(text):
    return {
        x.lower()
        for x in re.findall(
            r"\bT\d+[A-Za-z]?\b",
            str(text or ""),
            re.I,
        )
    }


def _result_contains_identifier(result, identifier):
    identifier = str(identifier).lower().strip()

    structured = result.get("identifiers", [])
    if isinstance(structured, str):
        structured = [structured]

    if identifier in {
        str(x).lower().strip()
        for x in structured
    }:
        return True

    metadata = result.get("metadata", {})

    fields = [
        result.get("title", ""),
        metadata.get("title", ""),
        result.get("text", ""),
        result.get("content", ""),
        result.get("chunk_text", ""),
        result.get("page_content", ""),
        result.get("chunk_id", ""),
    ]

    pattern = rf"(?<![a-z0-9]){re.escape(identifier)}(?![a-z0-9])"

    return any(
        re.search(pattern, str(field), re.I)
        for field in fields
    )


def _evidence_score(result):
    return (
        _safe_float(result.get("final_score"))
        + 0.05 * _safe_float(result.get("authority_score"))
        + 0.05 * _safe_float(result.get("reranker_score"))
        + 0.02 * _safe_float(result.get("domain_score"))
    )


def _strong_evidence(result):
    return (
        _safe_float(result.get("final_score")) >= 8.0
        or _safe_float(result.get("reranker_score")) >= 4.0
        or _safe_float(result.get("domain_score")) >= 20.0
        or _safe_float(result.get("authority_score")) >= 100.0
    )


def hallucination_guard(
    query,
    results,
    requested_identifiers=None,
    min_evidence=1,
    ambiguity_margin=0.15,
):
    requested = {
        str(x).lower().strip()
        for x in (requested_identifiers or [])
        if x
    }

    if not results:
        return {
            "status": "UNSUPPORTED",
            "reason": "No evidence was retrieved for the query.",
            "identifiers": sorted(requested),
            "covered_identifiers": [],
            "missing_identifiers": sorted(requested),
            "primary_section": None,
            "relevant_sections": [],
            "evidence_records": 0,
        }

    ranked = sorted(
        results,
        key=_evidence_score,
        reverse=True,
    )

    primary_section = _get_section(ranked[0])

    relevant_sections = []
    for result in ranked:
        section = _get_section(result)
        if section and section not in relevant_sections:
            relevant_sections.append(section)
        if len(relevant_sections) >= 5:
            break

    covered = set()

    for identifier in requested:
        if any(
            _result_contains_identifier(result, identifier)
            for result in ranked
        ):
            covered.add(identifier)

    missing = sorted(requested - covered)

    # HARD FAIL.
    if requested and missing:
        return {
            "status": "UNSUPPORTED",
            "reason": (
                "The query explicitly requests identifier(s) "
                f"{sorted(requested)}, but evidence is missing for "
                f"{missing}. Semantic similarity cannot substitute "
                "for identifier-specific evidence."
            ),
            "identifiers": sorted(requested),
            "covered_identifiers": sorted(covered),
            "missing_identifiers": missing,
            "primary_section": primary_section,
            "relevant_sections": relevant_sections,
            "evidence_records": len(results),
        }

    if requested:
        return {
            "status": "SUPPORTED",
            "reason": (
                "Evidence explicitly supports all requested identifiers."
            ),
            "identifiers": sorted(requested),
            "covered_identifiers": sorted(covered),
            "missing_identifiers": [],
            "primary_section": primary_section,
            "relevant_sections": relevant_sections,
            "evidence_records": len(results),
        }

    strong = [
        result for result in ranked
        if _strong_evidence(result)
    ]

    if len(strong) < min_evidence:
        return {
            "status": "UNSUPPORTED",
            "reason": (
                "Retrieved evidence is too weak or unrelated "
                "to reliably answer the conceptual query."
            ),
            "identifiers": [],
            "covered_identifiers": [],
            "missing_identifiers": [],
            "primary_section": primary_section,
            "relevant_sections": relevant_sections,
            "evidence_records": len(results),
        }

    return {
        "status": "SUPPORTED",
        "reason": (
            "The highest-ranked evidence sufficiently supports "
            "the conceptual query."
        ),
        "identifiers": [],
        "covered_identifiers": [],
        "missing_identifiers": [],
        "primary_section": primary_section,
        "relevant_sections": relevant_sections,
        "evidence_records": len(results),
    }


def validate_evidence(
    query,
    results,
    requested_identifiers=None,
):
    if requested_identifiers is None:
        requested_identifiers = [
            x.lower()
            for x in re.findall(
                r"\bT\d+[A-Za-z]*\b",
                str(query),
                re.I,
            )
        ]

    return hallucination_guard(
        query=query,
        results=results,
        requested_identifiers=requested_identifiers,
    )


def print_validation_result(result):
    print("\n" + "=" * 80)
    print("HALLUCINATION GUARD")
    print("=" * 80)
    print(f"Status: {result.get('status')}")
    print(f"Reason: {result.get('reason')}")
    print(f"Primary section: {result.get('primary_section')}")
    print(f"Relevant sections: {result.get('relevant_sections', [])}")
    print(f"Evidence records: {result.get('evidence_records', 0)}")
    print(f"Identifiers: {result.get('identifiers', [])}")
    print(f"Covered identifiers: {result.get('covered_identifiers', [])}")
    print(f"Missing identifiers: {result.get('missing_identifiers', [])}")
    print("=" * 80)
