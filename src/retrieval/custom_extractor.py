"""
3GPP Custom Query Extractor

Extracts telecom-specific concepts from natural-language queries.

The extractor identifies:

1. Timers
   - T300
   - T311
   - T304

2. RRC procedures
   - RRC connection establishment
   - RRC connection re-establishment
   - RRC connection resume
   - RRC connection release

3. Failure concepts
   - failure
   - expiry
   - timeout
   - radio link failure
   - handover failure

4. Technical identifiers
   - RRCReconfiguration
   - RRCSetup
   - connEstFailCount
   - Qoffsettemp

This module is intentionally independent from the retrieval engine.
"""


import re


# ============================================================
# KNOWN 3GPP PROCEDURES
# ============================================================

PROCEDURES = {

    "rrc_connection_establishment": [
        r"\brrc\s+connection\s+establishment\b",
        r"\brrc\s+establishment\b",
    ],

    "rrc_connection_reestablishment": [
        r"\brrc\s+connection\s+re[- ]establishment\b",
        r"\brrc\s+re[- ]establishment\b",
    ],

    "rrc_connection_resume": [
        r"\brrc\s+connection\s+resume\b",
    ],

    "rrc_connection_release": [
        r"\brrc\s+connection\s+release\b",
    ],

    "radio_link_failure": [
        r"\bradio\s+link\s+failure\b",
        r"\brlf\b",
    ],

    "handover_failure": [
        r"\bhandover\s+failure\b",
    ],

    "connection_establishment_failure": [
        r"\bconnection\s+establishment\s+failure\b",
        r"\bconnection\s+establishment\s+fail(?:ure|ed)\b",
    ],

    "security_mode": [
        r"\bsecurity\s+mode\s+command\b",
        r"\bsecurity\s+mode\b",
    ],
}


# ============================================================
# FAILURE / EVENT TERMS
# ============================================================

EVENT_TERMS = {

    "failure": [
        r"\bfailure\b",
        r"\bfailed\b",
        r"\bfail\b",
    ],

    "expiry": [
        r"\bexpiry\b",
        r"\bexpires\b",
        r"\bexpire\b",
        r"\bexpired\b",
    ],

    "timeout": [
        r"\btimeout\b",
        r"\btime[- ]out\b",
    ],

    "rejection": [
        r"\breject\b",
        r"\brejected\b",
        r"\brejection\b",
    ],

    "release": [
        r"\brelease\b",
        r"\breleased\b",
    ],

    "reestablishment": [
        r"\bre[- ]establishment\b",
        r"\breestablishment\b",
    ],
}


# ============================================================
# COMMON 3GPP TECHNICAL IDENTIFIERS
# ============================================================

TECHNICAL_IDENTIFIER_PATTERNS = [

    # RRC message/procedure identifiers
    r"\bRRC[A-Za-z0-9]+\b",

    # Timers
    r"\bT\d+[A-Za-z]*\b",

    # 3GPP-style parameter names
    r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b",

    # camelCase technical identifiers
    r"\b[a-z]+[A-Z][A-Za-z0-9]+\b",
]


# ============================================================
# TIMER EXTRACTION
# ============================================================

def extract_timers(query):
    """
    Extract timers such as:

        T300
        T301
        T311
        T304
    """

    matches = re.findall(
        r"\bT\d+[A-Za-z]*\b",
        query,
        flags=re.IGNORECASE,
    )

    timers = []

    for match in matches:

        timer = match.upper()

        if timer not in timers:
            timers.append(timer)

    return timers


# ============================================================
# PROCEDURE EXTRACTION
# ============================================================

def extract_procedures(query):
    """
    Extract known 3GPP procedures.

    Example:

        What is RRC connection establishment failure?

    returns:

        ["rrc_connection_establishment"]
    """

    query_lower = query.lower()

    procedures = []

    for procedure, patterns in PROCEDURES.items():

        for pattern in patterns:

            if re.search(
                pattern,
                query_lower,
                flags=re.IGNORECASE,
            ):

                if procedure not in procedures:
                    procedures.append(procedure)

                break

    return procedures


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_events(query):
    """
    Extract concepts such as:

        expiry
        failure
        timeout
        rejection
        release
    """

    query_lower = query.lower()

    events = []

    for event, patterns in EVENT_TERMS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                query_lower,
                flags=re.IGNORECASE,
            ):

                if event not in events:
                    events.append(event)

                break

    return events


# ============================================================
# TECHNICAL IDENTIFIER EXTRACTION
# ============================================================

def extract_technical_identifiers(query):
    """
    Extract technical identifiers while avoiding
    normal English words as much as possible.
    """

    identifiers = []

    for pattern in TECHNICAL_IDENTIFIER_PATTERNS:

        matches = re.findall(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        for match in matches:

            value = match.lower()

            # Avoid duplicate values
            if value not in identifiers:
                identifiers.append(value)

    return identifiers


# ============================================================
# NORMALIZE PROCEDURE
# ============================================================

def normalize_procedure_name(procedure):
    """
    Convert internal procedure name into
    human-readable terminology.
    """

    mapping = {

        "rrc_connection_establishment":
            "RRC connection establishment",

        "rrc_connection_reestablishment":
            "RRC connection re-establishment",

        "rrc_connection_resume":
            "RRC connection resume",

        "rrc_connection_release":
            "RRC connection release",

        "radio_link_failure":
            "radio link failure",

        "handover_failure":
            "handover failure",

        "connection_establishment_failure":
            "connection establishment failure",

        "security_mode":
            "security mode",
    }

    return mapping.get(
        procedure,
        procedure,
    )


# ============================================================
# MAIN CUSTOM EXTRACTION
# ============================================================

def extract_query_concepts(query):
    """
    Perform complete custom extraction.

    Returns:

    {
        "timers": [],
        "procedures": [],
        "events": [],
        "technical_identifiers": []
    }
    """

    timers = extract_timers(query)

    procedures = extract_procedures(query)

    events = extract_events(query)

    technical_identifiers = (
        extract_technical_identifiers(query)
    )

    return {

        "timers": timers,

        "procedures": procedures,

        "events": events,

        "technical_identifiers":
            technical_identifiers,
    }


# ============================================================
# DEBUG DISPLAY
# ============================================================

def print_extraction(query):
    """
    Pretty-print extraction results.
    """

    result = extract_query_concepts(query)

    print()
    print("=" * 70)
    print("3GPP CUSTOM QUERY EXTRACTION")
    print("=" * 70)

    print(f"Query: {query}")

    print()

    print("Timers:")
    print(result["timers"])

    print()

    print("Procedures:")

    for procedure in result["procedures"]:

        print(
            f"  - "
            f"{normalize_procedure_name(procedure)}"
        )

    print()

    print("Events:")
    print(result["events"])

    print()

    print("Technical identifiers:")
    print(result["technical_identifiers"])

    print("=" * 70)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "What is T300 expiry?",

        "What is T311 expiry?",

        "Compare T300 and T311 expiry.",

        "What is RRC connection establishment failure?",

        "What happens during radio link failure?",

        "Explain RRC connection re-establishment.",

        "What happens when RRCSetup is received?",
    ]

    for query in test_queries:

        print_extraction(query)