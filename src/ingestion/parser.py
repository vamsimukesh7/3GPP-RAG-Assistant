"""
3GPP TS 38.331 structure-aware DOCX parser.

Input:
    data/38331-j10.docx

Output:
    data/sections.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P


SECTION_NUMBER_RE = re.compile(
    r"^\d+(?:\.\d+)*[a-z]?\d*$"
)


def iter_block_items(parent: _Document) -> Iterator[Paragraph | Table]:
    """Read paragraphs and tables in original document order."""

    parent_elm = parent.element.body

    for child in parent_elm.iterchildren():

        if isinstance(child, CT_P):
            yield Paragraph(child, parent)

        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def clean_text(text: str) -> str:

    text = text.replace("\xa0", " ")

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def parse_heading_text(
    text: str
) -> Optional[Tuple[str, str]]:

    text = clean_text(text)

    if not text:
        return None

    # Example:
    # 5.3.3.7 T300 expiry

    match = re.match(
        r"^(?P<number>\d+(?:\.\d+)*[a-z]?\d*)\s+(?P<title>.+)$",
        text
    )

    if match:

        number = match.group("number")
        title = match.group("title")

        if SECTION_NUMBER_RE.fullmatch(number):

            # Remove trailing page number if present
            title = re.sub(
                r"\s+\d{1,4}$",
                "",
                title
            ).strip()

            return number, title

    return None


def build_toc_page_map(
    doc: Document
) -> Dict[str, Dict[str, object]]:

    page_map = {}

    for paragraph in doc.paragraphs:

        style = (
            paragraph.style.name.lower()
            if paragraph.style
            else ""
        )

        if not style.startswith("toc"):
            continue

        raw = paragraph.text.strip()

        parts = [
            x.strip()
            for x in raw.split("\t")
            if x.strip()
        ]

        if len(parts) >= 3:

            section_number = parts[0]

            if not SECTION_NUMBER_RE.fullmatch(
                section_number
            ):
                continue

            try:
                page = int(parts[-1])
            except ValueError:
                continue

            title = " ".join(parts[1:-1]).strip()

            page_map[section_number] = {
                "title": title,
                "page": page
            }

    return page_map


def table_to_text(table: Table) -> str:

    rows = []

    for row in table.rows:

        cells = [
            clean_text(cell.text)
            for cell in row.cells
        ]

        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    width = max(
        len(row)
        for row in rows
    )

    normalized = [
        row + [""] * (width - len(row))
        for row in rows
    ]

    lines = []

    # Header
    lines.append(
        " | ".join(normalized[0])
    )

    # Separator
    lines.append(
        " | ".join(["---"] * width)
    )

    # Rows
    for row in normalized[1:]:

        lines.append(
            " | ".join(row)
        )

    return "\n".join(lines)


def get_heading_level(
    paragraph: Paragraph
) -> Optional[int]:

    style = (
        paragraph.style.name
        if paragraph.style
        else ""
    )

    match = re.search(
        r"Heading\s+(\d+)",
        style,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def is_body_heading(
    paragraph: Paragraph
) -> bool:

    style = (
        paragraph.style.name
        if paragraph.style
        else ""
    )

    return bool(
        re.match(
            r"Heading\s+\d+",
            style,
            re.IGNORECASE
        )
    )


def extract_sections(
    doc: Document,
    page_map: Dict[str, Dict[str, object]]
) -> List[dict]:

    sections = []

    current = None

    def flush_current():

        nonlocal current

        if current is None:
            return

        parts = [
            x
            for x in current.pop("_parts")
            if x.strip()
        ]

        current["text"] = "\n\n".join(parts)

        current["character_count"] = len(
            current["text"]
        )

        current["word_count"] = len(
            current["text"].split()
        )

        sections.append(current)

        current = None

    for block in iter_block_items(doc):

        # -----------------------
        # Paragraph
        # -----------------------

        if isinstance(block, Paragraph):

            text = clean_text(
                block.text
            )

            if is_body_heading(block):

                parsed = parse_heading_text(
                    text
                )

                if parsed:

                    flush_current()

                    number, title = parsed

                    toc_meta = page_map.get(
                        number,
                        {}
                    )

                    current = {

                        "section": number,

                        "title": title,

                        "heading_level":
                            get_heading_level(
                                block
                            ),

                        "page":
                            toc_meta.get(
                                "page"
                            ),

                        "source": {

                            "specification":
                                "3GPP TS 38.331",

                            "version":
                                "V19.1.0",

                            "release":
                                "Release 19",

                            "filename":
                                "38331-j10.docx"
                        },

                        "_parts": []
                    }

                    continue

            if (
                current is not None
                and text
            ):

                current["_parts"].append(
                    text
                )

        # -----------------------
        # Table
        # -----------------------

        else:

            if current is not None:

                table_text = table_to_text(
                    block
                )

                if table_text:

                    current["_parts"].append(
                        table_text
                    )

    flush_current()

    return sections


def validate_sections(
    sections: List[dict]
) -> dict:

    numbers = [
        s["section"]
        for s in sections
    ]

    duplicates = sorted(
        {
            n
            for n in numbers
            if numbers.count(n) > 1
        }
    )

    return {

        "section_count":
            len(sections),

        "duplicate_section_numbers":
            duplicates,

        "sections_with_page":
            sum(
                s["page"] is not None
                for s in sections
            ),

        "sections_without_page":
            sum(
                s["page"] is None
                for s in sections
            ),

        "total_words":
            sum(
                s["word_count"]
                for s in sections
            )
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/38331-j10.docx"
    )

    parser.add_argument(
        "--output",
        default="data/sections.json"
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input DOCX not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading document...")

    doc = Document(
        str(input_path)
    )

    print("Reading table of contents...")

    page_map = build_toc_page_map(
        doc
    )

    print(
        f"TOC entries found: {len(page_map)}"
    )

    print("Extracting sections...")

    sections = extract_sections(
        doc,
        page_map
    )

    statistics = validate_sections(
        sections
    )

    payload = {

        "document": {

            "specification":
                "3GPP TS 38.331",

            "version":
                "V19.1.0",

            "release":
                "Release 19",

            "title":
                "NR; Radio Resource Control (RRC) protocol specification",

            "source_file":
                input_path.name
        },

        "statistics":
            statistics,

        "sections":
            sections
    }

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("========== SUCCESS ==========")
    print(
        json.dumps(
            statistics,
            indent=2
        )
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()