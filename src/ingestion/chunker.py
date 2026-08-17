"""
3GPP RAG - Structure-Aware Chunker

Input:
    data/sections.json

Output:
    data/chunks.json

The chunker:
- preserves 3GPP section numbers
- preserves titles and page numbers
- preserves parent section hierarchy
- splits large sections into smaller chunks
- adds overlap between chunks
- creates stable citation IDs
"""

import json
import re
from pathlib import Path


INPUT_FILE = Path("data/sections.json")
OUTPUT_FILE = Path("data/chunks.json")

# Approximate character limits.
# We will later tune these based on retrieval evaluation.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500


def clean_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph boundaries."""

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def split_into_paragraphs(text: str):
    """Split section text while preserving paragraph boundaries."""

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    return [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):
    """
    Create chunks using paragraph boundaries.

    We avoid cutting a paragraph in the middle whenever possible.
    """

    text = clean_text(text)

    if not text:
        return []

    paragraphs = split_into_paragraphs(text)

    chunks = []

    current = ""

    for paragraph in paragraphs:

        # If adding this paragraph stays within limit
        if len(current) + len(paragraph) + 2 <= chunk_size:

            if current:
                current += "\n\n"

            current += paragraph

        else:

            # Save current chunk
            if current:
                chunks.append(current)

            # If paragraph itself is too large,
            # split it safely by character length.
            if len(paragraph) > chunk_size:

                start = 0

                while start < len(paragraph):

                    end = start + chunk_size

                    piece = paragraph[start:end]

                    chunks.append(piece.strip())

                    start = end - overlap

                current = ""

            else:

                current = paragraph

    # Add final chunk
    if current:
        chunks.append(current)

    return chunks


def get_parent_section(section_number: str):
    """
    Example:

    5.3.3.7 -> 5.3.3
    5.3.3   -> 5.3
    5       -> None
    """

    parts = section_number.split(".")

    if len(parts) <= 1:
        return None

    return ".".join(parts[:-1])


def build_chunks(sections):

    all_chunks = []

    for section in sections:

        section_number = section["section"]
        title = section["title"]
        page = section["page"]
        text = section["text"]

        section_chunks = create_chunks(text)

        # Some headings contain no direct text.
        # We don't create empty chunks.
        if not section_chunks:
            continue

        parent = get_parent_section(
            section_number
        )

        for index, chunk_text in enumerate(
            section_chunks,
            start=1
        ):

            chunk_id = (
                f"3gpp_38_331_"
                f"{section_number}_"
                f"{index:03d}"
            )

            chunk = {

                "chunk_id": chunk_id,

                "text": chunk_text,

                "metadata": {

                    "specification":
                        "3GPP TS 38.331",

                    "version":
                        "V19.1.0",

                    "release":
                        "Release 19",

                    "section":
                        section_number,

                    "title":
                        title,

                    "parent_section":
                        parent,

                    "page":
                        page,

                    "chunk_index":
                        index,

                    "total_chunks_in_section":
                        len(section_chunks),

                    "source_file":
                        "38331-j10.docx"
                }
            }

            all_chunks.append(chunk)

    return all_chunks


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("Loading sections.json...")

    data = json.loads(
        INPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    sections = data["sections"]

    print(
        f"Sections loaded: {len(sections)}"
    )

    print("Creating RAG chunks...")

    chunks = build_chunks(
        sections
    )

    output = {

        "document": data["document"],

        "chunking": {

            "chunk_size":
                CHUNK_SIZE,

            "chunk_overlap":
                CHUNK_OVERLAP,

            "strategy":
                "structure-aware paragraph chunking"
        },

        "statistics": {

            "source_sections":
                len(sections),

            "total_chunks":
                len(chunks),

            "average_chunk_characters":
                round(
                    sum(
                        len(c["text"])
                        for c in chunks
                    ) / len(chunks),
                    2
                ) if chunks else 0
        },

        "chunks":
            chunks
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("========== SUCCESS ==========")

    print(
        f"Sections: {len(sections)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()