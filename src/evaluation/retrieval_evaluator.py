import json
import re
from pathlib import Path


class RetrievalEvaluator:

    def __init__(self, dataset_path, chunks_path):
        self.dataset_path = Path(dataset_path)
        self.chunks_path = Path(chunks_path)

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def normalize(self, text):
        return re.sub(r"\s+", " ", text.lower()).strip()

    def chunk_text(self, chunk):
        if isinstance(chunk, str):
            return chunk

        if isinstance(chunk, dict):
            return " ".join(
                str(chunk.get(key, ""))
                for key in [
                    "text",
                    "content",
                    "chunk",
                    "title",
                    "section_title"
                ]
            )

        return str(chunk)

    def search_chunks(self, query, top_k=5):

        query_words = set(
            re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
        )

        scored = []

        for chunk in self.chunks:

            text = self.normalize(self.chunk_text(chunk))

            score = 0

            for word in query_words:
                if word in text:
                    score += 1

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [chunk for score, chunk in scored[:top_k]]

    def evaluate_query(self, item, top_k=5):

        retrieved = self.search_chunks(
            item["question"],
            top_k=top_k
        )

        retrieved_text = "\n".join(
            self.normalize(self.chunk_text(chunk))
            for chunk in retrieved
        )

        expected_identifiers = item.get(
            "expected_identifiers", []
        )

        expected_terms = item.get(
            "expected_terms", []
        )

        identifier_hits = [
            identifier
            for identifier in expected_identifiers
            if self.normalize(identifier) in retrieved_text
        ]

        term_hits = [
            term
            for term in expected_terms
            if self.normalize(term) in retrieved_text
        ]

        identifier_score = (
            len(identifier_hits) / len(expected_identifiers)
            if expected_identifiers
            else 1.0
        )

        term_score = (
            len(term_hits) / len(expected_terms)
            if expected_terms
            else 1.0
        )

        return {
            "id": item["id"],
            "question": item["question"],
            "type": item["type"],
            "answerable": item["answerable"],
            "identifier_score": identifier_score,
            "term_score": term_score,
            "identifier_hits": identifier_hits,
            "term_hits": term_hits,
            "retrieved_count": len(retrieved)
        }

    def evaluate(self, top_k=5):

        results = []

        for item in self.dataset:
            result = self.evaluate_query(
                item,
                top_k=top_k
            )
            results.append(result)

        return results

    def calculate_metrics(self, results):

        if not results:
            return {}

        avg_identifier = sum(
            r["identifier_score"]
            for r in results
        ) / len(results)

        avg_terms = sum(
            r["term_score"]
            for r in results
        ) / len(results)

        successful = sum(
            1
            for r in results
            if r["identifier_score"] >= 0.5
            and r["term_score"] >= 0.5
        )

        retrieval_success_rate = (
            successful / len(results)
        )

        return {
            "total_queries": len(results),
            "average_identifier_score": round(
                avg_identifier, 4
            ),
            "average_term_score": round(
                avg_terms, 4
            ),
            "retrieval_success_rate": round(
                retrieval_success_rate, 4
            )
        }