import json
from pathlib import Path

from retrieval_evaluator import RetrievalEvaluator


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET = (
    BASE_DIR
    / "src"
    / "evaluation"
    / "evaluation_dataset.json"
)

CHUNKS = (
    BASE_DIR
    / "data"
    / "chunks.json"
)


def print_results(results):

    print("\n" + "=" * 70)
    print("3GPP RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for result in results:

        print(
            f"\n{result['id']} | "
            f"{result['question']}"
        )

        print(
            f"Identifier Score : "
            f"{result['identifier_score']:.2f}"
        )

        print(
            f"Term Score       : "
            f"{result['term_score']:.2f}"
        )

        print(
            f"Identifier Hits  : "
            f"{result['identifier_hits']}"
        )

        print(
            f"Term Hits        : "
            f"{result['term_hits']}"
        )


def main():

    evaluator = RetrievalEvaluator(
        dataset_path=DATASET,
        chunks_path=CHUNKS
    )

    results = evaluator.evaluate(
        top_k=5
    )

    metrics = evaluator.calculate_metrics(
        results
    )

    print_results(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Total Queries              : "
        f"{metrics['total_queries']}"
    )

    print(
        f"Average Identifier Score   : "
        f"{metrics['average_identifier_score']:.2%}"
    )

    print(
        f"Average Term Score         : "
        f"{metrics['average_term_score']:.2%}"
    )

    print(
        f"Retrieval Success Rate     : "
        f"{metrics['retrieval_success_rate']:.2%}"
    )

    output_path = (
        BASE_DIR
        / "src"
        / "evaluation"
        / "evaluation_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "metrics": metrics,
                "results": results
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()