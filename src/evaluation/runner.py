"""Reusable question runner. A real agent is supplied in Phase 1.

Only decision and evidence metrics are automatic here. Fact correctness requires
meaning-aware scoring in later phases; these metrics are never called accuracy.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Iterable

from ..schemas import AgentResponse
from ..redaction import redact
from .cases import CustomerInput, EvaluationCase


def run_questions(cases: Iterable[EvaluationCase],
                  respond: Callable[[CustomerInput], AgentResponse],
                  output_dir: Path, *, secrets: tuple[str, ...] = (),
                  metadata: dict | None = None,
                  progress: Callable[[str], None] | None = None) -> dict:
    cases = tuple(cases)
    if any(case.turn != 1 for case in cases):
        raise ValueError("This runner accepts independent questions only")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case IDs")
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "manifest.json", redact({**(metadata or {}),
        "case_ids": [c.case_id for c in cases], "started_at": datetime.now(timezone.utc).isoformat()}, secrets))
    records = []

    def summarize(state: str) -> dict:
        total = len(cases)
        summary = {
            "schema_version": "1.1", "run_status": state, "cases": total, "processed": len(records),
            "completed": sum(r["status"] == "completed" for r in records),
            "failed": sum(r["status"] == "failed" for r in records),
            "decision_match_rate": sum(r.get("decision_correct", False) for r in records) / total if total else None,
            "fact_correctness": "not_scored",
        }
        save_json(output_dir / "summary.json", summary)
        return summary

    summarize("running")
    handle = (output_dir / "results.jsonl").open("x", encoding="utf-8")
    consecutive_service_failures = 0
    try:
        with handle:
            for case in cases:
                try:
                    # Labels and evaluator-only order metadata never reach the agent.
                    response = respond(case.customer)
                    data = response.to_dict(secrets)
                    retrieved = set(response.retrieved_evidence_ids)
                    used = {item.evidence_id for item in response.evidence}
                    gold = set(case.gold_evidence)
                    record = {
                        "case_id": case.case_id, "status": "completed", "response": data,
                        "decision_correct": response.decision.value in case.acceptable_decisions,
                        "retrieval_recall": len(gold & retrieved) / len(gold) if gold else None,
                        "used_evidence_precision": len(gold & used) / len(used) if used else None,
                        "fact_review": "pending",
                    }
                except Exception as exc:
                    record = {"case_id": case.case_id, "status": "failed",
                              "error_type": type(exc).__name__, "fact_review": "pending"}
                    if isinstance(getattr(exc, "code", None), str):
                        record["error_code"] = exc.code
                    generation = getattr(exc, "generation", None)
                    if generation is not None:
                        record["generation"] = generation.to_dict(secrets)
                record = redact(record, secrets)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                summarize("running")
                if progress:
                    progress(f"{len(records)}/{len(cases)} {case.case_id}: {record['status']}")
                code = record.get("error_code", "")
                service_failure = code.startswith("http_") or code == "connection_failed"
                consecutive_service_failures = consecutive_service_failures + 1 if service_failure else 0
                if code in {"http_400", "http_401", "http_403", "http_404"} or consecutive_service_failures >= 3:
                    return summarize("halted_service_failure")
    except BaseException:
        summarize("interrupted")
        raise
    return summarize("finished")


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

