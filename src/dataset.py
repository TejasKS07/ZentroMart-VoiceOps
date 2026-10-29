"""Read-only public knowledge loading. Private order data stays behind tools.

All CSV values stay strings, including empty cells. Consumers explicitly parse
numbers/dates rather than accidentally converting identifiers or missing values.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys

from .schemas import Evidence

TABLE_KEYS = {
    "customers": ("customer_id",), "orders": ("order_id",),
    "order_items": ("order_id", "line_no"), "products": ("sku",),
    "pincodes": ("pincode",), "holidays": ("date",), "support_logs": ("log_id",),
}


class DatasetError(ValueError):
    pass


def read_csv(path: Path, required: tuple[str, ...] = ()) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetError(f"Missing dataset file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers or len(headers) != len(set(headers)) or set(required) - set(headers):
            raise DatasetError(f"Invalid/missing CSV columns: {path.name}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise DatasetError(f"Malformed CSV row: {path.name}")
    return rows


def indexed(rows: list[dict[str, str]], keys: tuple[str, ...], name: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if not all(key) or key in result:
            raise DatasetError(f"Blank/duplicate primary key: {name}")
        result[key] = row
    return result


@dataclass(frozen=True)
class KnowledgeDocument:
    source: str
    metadata: dict[str, str]
    sections: tuple[Evidence, ...]


def read_document(path: Path) -> KnowledgeDocument:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
    if len(parts) != 3 or parts[0].strip():
        raise DatasetError(f"Missing document header: {path.name}")
    metadata = {}
    for line in parts[1].splitlines():
        if line.strip():
            key, separator, value = line.partition(":")
            if not separator:
                raise DatasetError(f"Invalid document header: {path.name}")
            metadata[key.strip()] = value.strip()
    required = {"doc_id", "title", "doc_type", "effective_from", "status"}
    if not required <= metadata.keys():
        raise DatasetError(f"Incomplete document metadata: {path.name}")
    headings = list(re.finditer(r"^##\s+\[([^\]]+)\].*$", parts[2], re.MULTILINE))
    sections = tuple(Evidence(
        evidence_id=match.group(1), source=f"kb/{path.name}",
        text=parts[2][match.start():headings[i+1].start() if i+1 < len(headings) else None].strip(),
        metadata=dict(metadata),
    ) for i, match in enumerate(headings))
    if not sections:
        raise DatasetError(f"No section IDs: {path.name}")
    return KnowledgeDocument(f"kb/{path.name}", metadata, sections)


@dataclass(frozen=True, repr=False)
class PublicKnowledge:
    products: dict[str, dict[str, str]]
    pincodes: dict[str, dict[str, str]]
    holidays: tuple[dict[str, str], ...]
    documents: tuple[KnowledgeDocument, ...]

    @classmethod
    def load(cls, root: Path) -> "PublicKnowledge":
        products = read_csv(root / "data/products.csv", ("sku", "return_category"))
        pincodes = read_csv(root / "data/pincodes.csv", ("pincode", "serviceable"))
        holidays = read_csv(root / "data/holidays.csv", ("date", "name"))
        indexed(holidays, ("date",), "holidays")
        files = sorted((root / "kb").glob("*.md"))
        if not files:
            raise DatasetError("No knowledge-base documents found")
        documents = tuple(read_document(path) for path in files)
        ids = [section.evidence_id for doc in documents for section in doc.sections]
        if len(ids) != len(set(ids)):
            raise DatasetError("Duplicate evidence section IDs")
        return cls(
            {k[0]: v for k, v in indexed(products, ("sku",), "products").items()},
            {k[0]: v for k, v in indexed(pincodes, ("pincode",), "pincodes").items()},
            tuple(holidays), documents,
        )


def inventory(root: Path) -> dict[str, int]:
    """Validate keys and expose counts, never records, for private tables."""
    counts = {}
    for table, keys in TABLE_KEYS.items():
        rows = read_csv(root / "data" / f"{table}.csv", keys)
        indexed(rows, keys, table)
        counts[table] = len(rows)
    counts["kb_documents"] = len(PublicKnowledge.load(root).documents)
    return counts


def validate_supplied_dataset(root: Path) -> None:
    script = root / "validate_dataset.py"
    if not script.is_file():
        raise DatasetError("Missing supplied validate_dataset.py")
    result = subprocess.run([sys.executable, "-B", str(script)], cwd=root,
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        # The supplied validator may print data. Do not forward its output.
        raise DatasetError("Supplied dataset validator failed; run it directly for diagnostics")

