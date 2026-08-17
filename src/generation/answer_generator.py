"""
3GPP RAG - Grounded Answer Generator

Pipeline:

    Retrieval Results
          ↓
    Evidence Guard
          ↓
    SUPPORTED?
       ↓       ↓
      YES      NO
       ↓       ↓
    Groq LLM   Block
       ↓
    Grounded Answer
       ↓
    Citations
       ↓
    Final Answer
"""

from typing import List, Dict, Optional

# ============================================================
# IMPORTANT:
# llm_generator.py is located in the project root.
#
# We import it using an absolute project-level import.
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# from llm_generator import generate_answer

from src.llm.local_llm import generate_answer

# ============================================================
# CONFIGURATION
# ============================================================

UNSUPPORTED_MESSAGE = (
    "The retrieved 3GPP evidence is insufficient to answer "
    "this question."
)


# ============================================================
# EVIDENCE STATUS
# ============================================================

def get_evidence_status(
    validation_result: Dict
) -> str:

    if not validation_result:
        return "UNSUPPORTED"

    return str(
        validation_result.get(
            "status",
            "UNSUPPORTED"
        )
    ).upper()


def is_generation_allowed(
    validation_result: Dict
) -> bool:

    return (
        get_evidence_status(
            validation_result
        ) == "SUPPORTED"
    )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: List[Dict]
) -> str:

    if not results:
        return ""

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        section = metadata.get(
            "section",
            ""
        )

        title = metadata.get(
            "title",
            ""
        )

        page = metadata.get(
            "page",
            ""
        )

        chunk_id = result.get(
            "chunk_id",
            ""
        )

        text = result.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
Evidence {i}
Section: {section}
Title: {title}
Page: {page}
Chunk ID: {chunk_id}

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# BUILD GROQ PROMPT
# ============================================================

def build_prompt(
    query: str,
    results: List[Dict]
) -> str:

    context = build_context(
        results
    )

    return f"""
You are a strict 3GPP technical assistant.

Your task is to answer the user's question using ONLY
the supplied verified 3GPP evidence.

The supplied evidence is your ONLY source of truth.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Do not guess missing information.

4. Preserve exact 3GPP terminology.

5. Every technical claim must be supported by
   the supplied evidence.

6. Mention the relevant 3GPP section and page
   for important technical claims.

7. If the question contains a specific identifier
   such as T300, T301, T310, T311, T320, etc.,
   use only evidence explicitly referring to that
   identifier.

8. Never substitute evidence about another timer
   for the requested timer.

9. Never assume that two procedures are equivalent.

10. If evidence supports only part of the question,
    answer only that supported part.

11. If the evidence does not establish an answer,
    say exactly:

    "The retrieved 3GPP evidence is insufficient
    to answer this question."

12. Do not mention information that is not present
    in the evidence.

13. Keep the answer concise and technically precise.

14. Do not cite evidence that does not support
    the claim being made.

15. When possible, place the citation directly
    after the supported statement using:

    [3GPP TS 38.331, Section X.X.X, Page Y]

USER QUESTION:

{query}

VERIFIED 3GPP EVIDENCE:

{context}

ANSWER:
"""


# ============================================================
# UNSUPPORTED RESPONSE
# ============================================================

def build_unsupported_response(
    query: str,
    validation_result: Dict
) -> Dict:

    missing = validation_result.get(
        "missing_identifiers",
        []
    )

    primary_section = validation_result.get(
        "primary_section"
    )

    relevant_sections = validation_result.get(
        "relevant_sections",
        []
    )

    reason = validation_result.get(
        "reason",
        UNSUPPORTED_MESSAGE
    )

    if missing:

        message = (
            "I cannot provide a verified answer because "
            "the retrieved 3GPP evidence does not contain "
            f"supporting evidence for: {missing}. "
            "Evidence about other timers or procedures "
            "cannot be substituted."
        )

    else:

        message = UNSUPPORTED_MESSAGE

    return {
        "status": "UNSUPPORTED",
        "query": query,
        "answer": message,
        "reason": reason,
        "primary_section": primary_section,
        "relevant_sections": relevant_sections,
        "citations": [],
        "evidence_count": validation_result.get(
            "evidence_records",
            0
        ),
        "llm_generation_allowed": False
    }


# ============================================================
# FORMAT CITATIONS
# ============================================================

def format_citations(
    results: List[Dict]
) -> List[Dict]:

    citations = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        citations.append(
            {
                "section": metadata.get(
                    "section"
                ),
                "title": metadata.get(
                    "title"
                ),
                "page": metadata.get(
                    "page"
                ),
                "chunk_id": result.get(
                    "chunk_id"
                )
            }
        )

    return citations


# ============================================================
# PREPARE SUPPORTED ANSWER
# ============================================================

def prepare_supported_answer_request(
    query: str,
    results: List[Dict],
    validation_result: Dict
) -> Dict:

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not is_generation_allowed(
        validation_result
    ):

        return build_unsupported_response(
            query,
            validation_result
        )

    # --------------------------------------------------------
    # NO EVIDENCE = BLOCK
    # --------------------------------------------------------

    if not results:

        return {
            "status": "UNSUPPORTED",
            "query": query,
            "answer": UNSUPPORTED_MESSAGE,
            "citations": [],
            "evidence_count": 0,
            "llm_generation_allowed": False
        }

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = build_prompt(
        query,
        results
    )

    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            prompt
        )

    except Exception as e:

        return {
            "status": "ERROR",
            "query": query,
            "answer": (
                "Answer generation failed. "
                "The verified evidence was available, "
                "but the LLM could not generate a response."
            ),
            "error": str(e),
            "prompt": prompt,
            "citations": format_citations(
                results
            ),
            "evidence_count": len(results),
            "llm_generation_allowed": True
        }

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {
        "status": "SUPPORTED",
        "query": query,
        "answer": answer,
        "prompt": prompt,
        "citations": format_citations(
            results
        ),
        "evidence_count": len(results),
        "primary_section": validation_result.get(
            "primary_section"
        ),
        "relevant_sections": validation_result.get(
            "relevant_sections",
            []
        ),
        "llm_generation_allowed": True
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def prepare_answer_request(
    query: str,
    results: List[Dict],
    validation_result: Optional[Dict] = None
) -> Dict:

    # --------------------------------------------------------
    # NEVER GENERATE WITHOUT VALIDATION
    # --------------------------------------------------------

    if validation_result is None:

        return {
            "status": "UNSUPPORTED",
            "query": query,
            "answer": (
                "The evidence validation result "
                "is missing. LLM generation has "
                "been blocked for safety."
            ),
            "prompt": "",
            "citations": [],
            "evidence_count": len(results),
            "llm_generation_allowed": False
        }

    # --------------------------------------------------------
    # SUPPORTED → GROQ
    # UNSUPPORTED → BLOCK
    # --------------------------------------------------------

    return prepare_supported_answer_request(
        query,
        results,
        validation_result
    )