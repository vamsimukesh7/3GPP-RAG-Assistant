import re


class AnswerEvaluator:

    def normalize(self, text):
        return re.sub(
            r"\s+",
            " ",
            text.lower()
        ).strip()

    def evaluate(
        self,
        question,
        answer,
        evidence,
        expected_identifiers=None
    ):

        expected_identifiers = expected_identifiers or []

        answer_normalized = self.normalize(answer)
        evidence_normalized = self.normalize(evidence)

        identifier_results = {}

        for identifier in expected_identifiers:

            identifier_normalized = self.normalize(
                identifier
            )

            identifier_results[identifier] = {
                "in_answer":
                    identifier_normalized in answer_normalized,

                "in_evidence":
                    identifier_normalized in evidence_normalized
            }

        supported_sentences = 0
        total_sentences = 0

        sentences = re.split(
            r"[.!?]+",
            answer
        )

        evidence_words = set(
            re.findall(
                r"\b[a-zA-Z0-9_-]+\b",
                evidence_normalized
            )
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            total_sentences += 1

            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z0-9_-]+\b",
                    self.normalize(sentence)
                )
            )

            if not sentence_words:
                continue

            overlap = (
                len(sentence_words & evidence_words)
                / len(sentence_words)
            )

            if overlap >= 0.20:
                supported_sentences += 1

        faithfulness = (
            supported_sentences / total_sentences
            if total_sentences
            else 0
        )

        return {
            "question": question,
            "faithfulness_score": round(
                faithfulness,
                4
            ),
            "supported_sentences": supported_sentences,
            "total_sentences": total_sentences,
            "identifiers": identifier_results
        }