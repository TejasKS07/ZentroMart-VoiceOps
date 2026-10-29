"""Offline retrieval metrics. Gold labels stay outside indexing/search code."""

from collections import defaultdict
from pathlib import Path
import json

from .cases import load_questions
from .runner import save_json


def evaluate_retrieval(index, root: Path, output: Path, *, k: int = 8, asr: bool = False) -> dict:
    rows = []
    by_language = defaultdict(list)
    for case in load_questions(root, asr=asr):
        result = index.search(case.customer.text, k=k)
        retrieved = {hit.evidence.evidence_id for hit in result.candidates}
        context = {e.evidence_id for e in result.context}
        gold = set(case.gold_evidence)
        row = {"case_id": case.case_id, "language": case.language,
               "candidate_ids": [hit.evidence.evidence_id for hit in result.candidates],
               "context_ids": [e.evidence_id for e in result.context],
               "gold_ids": sorted(gold), "missing_ids": sorted(gold-context),
               "recall_at_k": len(gold & retrieved)/len(gold) if gold else None,
               "context_recall": len(gold & context)/len(gold) if gold else None,
               "precision_at_k": len(gold & retrieved)/len(retrieved) if gold and retrieved else None,
               "context_precision": len(gold & context)/len(context) if gold and context else None}
        rows.append(row)
        if gold:
            by_language[case.language].append(row)
    answerable = [r for r in rows if r["recall_at_k"] is not None]
    summary = {"mode": "local_retrieval_only", "variant": "asr" if asr else "original", "k": k,
               "embedding_model": index.embedder.name, "index_fingerprint": index.fingerprint,
               "questions": len(rows), "questions_with_gold": len(answerable),
               "mean_recall_at_k": sum(r["recall_at_k"] for r in answerable)/len(answerable),
               "mean_context_recall": sum(r["context_recall"] for r in answerable)/len(answerable),
               "mean_precision_at_k": sum(r["precision_at_k"] for r in answerable)/len(answerable),
               "mean_context_precision": sum(r["context_precision"] for r in answerable)/len(answerable),
               "mean_context_chunks": sum(len(r["context_ids"]) for r in rows)/len(rows),
               "per_language": {language: {"questions": len(items), "context_recall": sum(r["context_recall"] for r in items)/len(items)} for language, items in by_language.items()},
               "generation_accuracy": "not_run_deferred_until_project_complete", "gemini_calls": 0}
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    save_json(output / "summary.json", summary)
    return summary
