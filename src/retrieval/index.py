"""Section-preserving local vector index, exact catalog lookup and provenance."""

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from collections import Counter
import numpy as np

from ..config import REFERENCE_DATE
from ..dataset import PublicKnowledge
from ..schemas import Evidence

AUTHORITY = {"meta": 0, "policy": 1, "catalog": 1, "agent_guide": 2, "help": 3, "faq": 4, "marketing": 5}
INDEX_VERSION = 3


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted([*(root / "kb").glob("*.md"), root / "data/products.csv", root / "data/pincodes.csv", root / "data/holidays.csv"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:-[^\W_]+)*", text.casefold(), flags=re.UNICODE)


def public_skus(query: str, catalog: tuple[Evidence, ...]) -> tuple[set[str], set[str]]:
    """Resolve public catalog candidates only; never normalize private order IDs.

    ASR homophones produce explicitly uncertain candidates, requiring caller
    confirmation before any decision depending on that exact SKU.
    """
    exact = set(re.findall(r"\b[A-Z]{2,8}-\d{4}\b", query.upper()))
    uncertain = set()
    words = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "to": "2", "too": "2"}
    prefixes = {e.metadata["sku"].split("-")[0] for e in catalog}
    for prefix in prefixes:
        for match in re.finditer(r"\b" + re.escape(prefix) + r"[\s-]+([^\n]+)", query, re.I):
            parts = re.findall(r"[a-z]+|\d+", match[1].lower())
            digits = ""
            ambiguous = False
            for part in parts:
                if part.isdigit():
                    digits += part
                elif part in words:
                    digits += words[part]
                    ambiguous |= part in {"to", "too", "oh"}
                else:
                    break
                if len(digits) >= 4:
                    break
            if len(digits) == 4:
                sku = prefix + "-" + digits
                (uncertain if ambiguous else exact).add(sku)
    return exact, uncertain


def delivery_date_from_query(query: str) -> date | None:
    match = re.search(r"deliver(?:ed|y)\s+(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(20\d{2}))?", query, re.I)
    if match:
        for format in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(f"{match[1]} {match[2]} {match[3] or REFERENCE_DATE.year}", format).date()
            except ValueError:
                continue
    return None


def chunks(knowledge: PublicKnowledge) -> tuple[Evidence, ...]:
    result = [section for doc in knowledge.documents for section in doc.sections]
    for sku, row in sorted(knowledge.products.items()):
        result.append(Evidence(f"products.csv:{sku}", "data/products.csv",
            "\n".join(f"{key}: {value}" for key, value in row.items()),
            {"doc_type": "catalog", "status": "current", "effective_from": REFERENCE_DATE.isoformat(), "sku": sku,
             "title": row["title"], "return_category": row["return_category"]}))
    return tuple(result)


def eligible(evidence: Evidence, *, delivered_on: date | None, reference_date: date) -> bool:
    metadata = evidence.metadata
    if date.fromisoformat(metadata["effective_from"]) > reference_date:
        return False
    if metadata["status"] != "superseded":
        return True
    # The successor must explicitly authorize the old version; handled by index.
    return False


@dataclass(frozen=True)
class SearchHit:
    evidence: Evidence
    similarity: float
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    candidates: tuple[SearchHit, ...]
    context: tuple[Evidence, ...]


class KnowledgeIndex:
    def __init__(self, evidence: tuple[Evidence, ...], vectors: np.ndarray, embedder, fingerprint: str):
        self.evidence = evidence
        self.embedder = embedder
        self.fingerprint = fingerprint
        if vectors.ndim != 2 or len(vectors) != len(evidence) or not np.isfinite(vectors).all():
            raise ValueError("Invalid index vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero index vector")
        self.vectors = vectors / norms
        self.by_id = {e.evidence_id: e for e in evidence}
        if len(self.by_id) != len(evidence):
            raise ValueError("Duplicate index evidence IDs")
        self.term_counts = [Counter(tokens(e.metadata.get("title", "") + " " + e.text)) for e in evidence]
        self.document_frequency = Counter(term for counts in self.term_counts for term in counts)

    @classmethod
    def build(cls, root: Path, embedder) -> "KnowledgeIndex":
        evidence = chunks(PublicKnowledge.load(root))
        texts = [e.metadata.get("title", "") + "\n" + e.text for e in evidence]
        return cls(evidence, embedder.passages(texts), embedder, source_fingerprint(root))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors, allow_pickle=False)
        payload = {"version": INDEX_VERSION, "model": self.embedder.name, "fingerprint": self.fingerprint,
                   "embedding_fingerprint": getattr(self.embedder, "fingerprint", self.embedder.name),
                   "evidence": [asdict(e) for e in self.evidence]}
        (path / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedder, root: Path) -> "KnowledgeIndex":
        data = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if data["version"] != INDEX_VERSION or data["model"] != embedder.name:
            raise ValueError("Embedding model/index version changed; rebuild the index")
        if data["embedding_fingerprint"] != getattr(embedder, "fingerprint", embedder.name):
            raise ValueError("Embedding weights/tokenizer changed; rebuild the index")
        if data["fingerprint"] != source_fingerprint(root):
            raise ValueError("Knowledge sources changed; rebuild the index")
        return cls(tuple(Evidence(**item) for item in data["evidence"]),
                   np.load(path / "vectors.npy", allow_pickle=False), embedder, data["fingerprint"])

    def _historical_allowed(self, evidence: Evidence, delivered_on: date | None) -> bool:
        if delivered_on is None:
            return False
        old_id = evidence.metadata.get("doc_id")
        successors = [e for e in self.evidence if e.metadata.get("supersedes") == old_id and e.metadata.get("status") == "current"]
        # Retain old sections only where a current successor explicitly preserves
        # a previous delivery-date window. No use based on the old effective date alone.
        old_version = evidence.metadata.get("version", "").split(".")[0]
        return any("delivered before" in e.text.lower() and f"version {old_version}" in e.text.lower()
                   and delivered_on < date.fromisoformat(e.metadata["effective_from"]) for e in successors)

    def search(self, query: str, *, k: int = 8, previous_topic: str = "",
               delivered_on: date | None = None) -> RetrievalResult:
        if not query.strip() or not 1 <= k <= 30:
            raise ValueError("Search needs a nonempty query and k between 1 and 30")
        effective_query = (previous_topic.strip() + "\n" + query.strip()).strip() if previous_topic else query.strip()
        delivered_on = delivered_on or delivery_date_from_query(effective_query)
        query_vector = self.embedder.queries([effective_query])[0]
        norm = np.linalg.norm(query_vector)
        if not np.isfinite(query_vector).all() or not norm:
            raise ValueError("Invalid query embedding")
        similarities = self.vectors @ (query_vector / norm)
        terms = set(tokens(effective_query))
        catalog = tuple(e for e in self.evidence if e.metadata.get("doc_type") == "catalog")
        exact_skus, uncertain_skus = public_skus(effective_query, catalog)
        sku_matches = exact_skus | uncertain_skus
        hits = []
        for i, evidence in enumerate(self.evidence):
            if not eligible(evidence, delivered_on=delivered_on, reference_date=REFERENCE_DATE):
                if evidence.metadata.get("status") != "superseded" or not self._historical_allowed(evidence, delivered_on):
                    continue
            title_match = 0.0
            if evidence.metadata.get("doc_type") == "catalog":
                title_terms = set(tokens(evidence.metadata.get("title", ""))) - {"of", "and", "the", "pack"}
                shared = len(title_terms & terms)
                title_match = shared / max(1, len(title_terms))
                if evidence.metadata.get("sku") not in sku_matches and not (shared >= 3 and title_match >= 0.6):
                    continue
            overlap = sum(math.log(1+len(self.evidence)/(1+self.document_frequency[t])) for t in terms if t in self.term_counts[i])
            lexical = overlap / max(1, len(terms))
            authority = AUTHORITY.get(evidence.metadata.get("doc_type"), 5)
            score = float(similarities[i]) + 0.025 * lexical + 0.012 * (5-authority)
            if evidence.metadata.get("sku") in sku_matches:
                score += 1
            elif title_match:
                score += 0.7 * title_match
            if evidence.metadata.get("sku") in uncertain_skus:
                evidence = Evidence(evidence.evidence_id, evidence.source, evidence.text,
                                    {**evidence.metadata, "uncertain_asr_candidate": "true"})
            hits.append(SearchHit(evidence, float(similarities[i]), score))
        hits.sort(key=lambda hit: (-hit.score, hit.evidence.evidence_id))
        candidates = tuple(hits[:k])
        selected = {hit.evidence.evidence_id: hit.evidence for hit in candidates}
        # Include the source-of-truth rules and explicit cross-references, retaining
        # original chunks instead of replacing facts with hand-authored summaries.
        for evidence in self.evidence:
            if evidence.metadata.get("doc_type") in {"meta", "agent_guide"}:
                selected[evidence.evidence_id] = evidence
        # Adjacent policy sections carry scope, exceptions and fee/timeline details
        # that short isolated sections omit. Include neighbors from the same doc.
        for evidence in tuple(selected.values()):
            if evidence.metadata.get("doc_type") == "policy":
                siblings = [e for e in self.evidence if e.source == evidence.source]
                position = next(i for i, e in enumerate(siblings) if e.evidence_id == evidence.evidence_id)
                for sibling in siblings[max(0, position-1):position+2]:
                    if eligible(sibling, delivered_on=delivered_on, reference_date=REFERENCE_DATE):
                        selected[sibling.evidence_id] = sibling
        for evidence in tuple(selected.values()):
            references = re.findall(r"\b[A-Z]{2,5}(?:v\d+)?-\d+(?:\.\d+)?\b", evidence.text)
            for reference in references:
                linked = self.by_id.get(reference)
                if linked and eligible(linked, delivered_on=delivered_on, reference_date=REFERENCE_DATE):
                    selected[reference] = linked
        # If a historical section is relevant, always include its current successor.
        for evidence in tuple(selected.values()):
            if evidence.metadata.get("status") == "superseded":
                for current in self.evidence:
                    if current.metadata.get("supersedes") == evidence.metadata.get("doc_id") and "delivered before" in current.text.lower():
                        selected[current.evidence_id] = current
        context = tuple(sorted(selected.values(), key=lambda e: (AUTHORITY.get(e.metadata.get("doc_type"), 5), e.evidence_id)))
        return RetrievalResult(effective_query, candidates, context)
