"""
3GPP RAG Chatbot - production test entry point.

Uses:
    FAISS + BM25
    exact identifier retrieval
    identifier-aware cross encoder
    hallucination/evidence guard
    verified answer generator
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_search import (
    load_resources,
    run_query,
    display_results,
)
from src.generation.answer_generator import prepare_answer_request


def main():
    print("=" * 80)
    print("3GPP TS 38.331 RAG CHATBOT")
    print("=" * 80)
    print("Loading retrieval resources...")

    resources = load_resources()

    print("\nReady.")
    print("Test:")
    print("  What happens when T311 expires?")
    print("  What happens when T300 expires?")
    print("  What happens when T999 expires?")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nAsk your 3GPP question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            results, validation_result = run_query(
                query,
                *resources,
            )

            display_results(
                query,
                results,
                validation_result,
            )

            print("\n" + "=" * 80)
            print("GENERATED ANSWER")
            print("=" * 80)

            # Only verified evidence reaches the answer generator.
            response = prepare_answer_request(
                query,
                results,
                validation_result,
            )

            print(f"Status: {response.get('status')}")
            print(f"\n{response.get('answer', '')}")

            if response.get("citations"):
                print("\nCITATIONS")
                for citation in response["citations"]:
                    print(
                        f"- Section {citation.get('section')} | "
                        f"{citation.get('title')} | "
                        f"Page {citation.get('page')} | "
                        f"{citation.get('chunk_id')}"
                    )

        except Exception as exc:
            print("\nERROR")
            print(type(exc).__name__, ":", exc)


if __name__ == "__main__":
    main()
