"""
Local Qwen 2.5 1.5B GGUF LLM.

Model:
qwen2.5-1.5b-instruct-q4_0.gguf

Runs completely locally using llama.cpp.
"""

from __future__ import annotations

from pathlib import Path

from llama_cpp import Llama


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen2.5-1.5b-instruct-q4_0.gguf"
)


class LocalQwen:

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
    ):

        model_path = Path(model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"Qwen model not found:\n"
                f"{model_path}"
            )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Loading local Qwen 2.5 1.5B..."
        )

        print(
            f"Model: {model_path}"
        )

        print(
            "=" * 70
        )

        self.llm = Llama(
            model_path=str(model_path),

            n_ctx=4096,

            n_threads=8,

            n_gpu_layers=0,

            verbose=False,
        )

        print(
            "Qwen model loaded successfully."
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> str:

        response = self.llm.create_chat_completion(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict technical "
                        "RAG assistant.\n\n"
                        "Answer ONLY using the supplied "
                        "context.\n"
                        "Do not use outside knowledge.\n"
                        "Do not invent facts.\n"
                        "If the context does not contain "
                        "the answer, say that the evidence "
                        "is insufficient."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=temperature,

            max_tokens=max_tokens,
        )

        return (
            response["choices"][0]["message"]
            ["content"]
            .strip()
        )


_local_qwen = None


def get_local_llm():

    global _local_qwen

    if _local_qwen is None:

        _local_qwen = LocalQwen()

    return _local_qwen


def generate_local_answer(
    prompt: str,
    max_tokens: int = 700,
) -> str:

    llm = get_local_llm()

    return llm.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )