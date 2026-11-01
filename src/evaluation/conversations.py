"""Ordered conversation replay; labels stay outside the agent callback."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from .cases import load_conversations
from .runner import save_json
from ..config import PROJECT_ROOT
from ..schemas import CustomerInput


def replay_conversations(dataset_dir: Path, output_dir: Path, agent, *, mode="offline", secrets=(), progress=None):
    if mode not in {"offline", "live"}:
        raise ValueError("Unknown conversation replay mode")
    if mode == "offline" and agent.rag is not None:
        raise ValueError("Offline replay must not have a generation agent attached")
    if output_dir.resolve().is_relative_to(dataset_dir.resolve()):
        raise ValueError("Evaluation output must be outside the dataset")
    groups = load_conversations(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    save_json(output_dir / "manifest.json", {
        "mode": mode, "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
        "conversations": len(groups), "turns": sum(map(len, groups.values())),
        "practice_sha256": hashlib.sha256((dataset_dir / "practice/conversations.csv").read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PROJECT_ROOT / "src").rglob("*")) if p.suffix in {".py", ".txt"}},
        "agent_inputs": "Customer message and unique session ID only; no evaluator metadata.",
        "fact_correctness": "not_scored", "gemini_calls": 0 if mode == "offline" else "see generation traces"})
    records = []
    consecutive_service_failures = 0

    def summarize(status):
        scenario_results = {}
        for group, cases in groups.items():
            turns = [r for r in records if r["conversation_id"] == group]
            scenario_results[group] = {"processed": len(turns), "turns": len(cases),
                "passed_routing": len(turns) == len(cases) and all(r.get("decision_correct") and r.get("tool_calls_correct") for r in turns)}
        total = sum(map(len, groups.values()))
        summary = {"run_status": status, "mode": mode, "conversations": len(groups), "turns": total,
            "processed": len(records), "failed": sum(r["status"] == "failed" for r in records),
            "decision_matches": sum(r.get("decision_correct", False) for r in records),
            "decision_match_rate": sum(r.get("decision_correct", False) for r in records) / total,
            "tool_call_matches": sum(r.get("tool_calls_correct", False) for r in records),
            "tool_call_match_rate": sum(r.get("tool_calls_correct", False) for r in records) / total,
            "scenarios_passing_routing": sum(r["passed_routing"] for r in scenario_results.values()),
            "scenarios": scenario_results, "fact_correctness": "not_scored",
            "note": "Routing and tool-name matches are not factual or generated-answer accuracy."}
        save_json(output_dir / "summary.json", summary)
        return summary

    summarize("running")
    try:
        with (output_dir / "results.jsonl").open("x", encoding="utf-8") as handle:
            for group, cases in groups.items():
                session = run_id + ":" + group
                # Fresh session avoids collisions with ticket ledgers from earlier runs.
                agent.clear_session(session)
                try:
                    for case in cases:
                        try:
                            response = agent.respond(CustomerInput(case.customer.text, session))
                            # Proof-bearing customer messages/expected arguments are never persisted.
                            data = agent.operations.gateway.sanitize(session, response.to_dict(secrets))
                            expected = Counter(re.findall(r"(\w+)\(", case.labels["expected_tool_calls"]))
                            actual = Counter(e.name for e in response.tool_events)
                            gold = set(case.gold_evidence)
                            used = {e.evidence_id for e in response.evidence}
                            retrieved = set(response.retrieved_evidence_ids)
                            record = {"case_id": case.case_id, "conversation_id": group, "turn": case.turn,
                                "status": "completed", "response": data,
                                "decision_correct": response.decision.value in case.acceptable_decisions,
                                "expected_tool_names": list(expected.elements()), "tool_calls_correct": actual == expected,
                                "retrieval_recall": len(gold & retrieved) / len(gold) if gold and retrieved else None,
                                "used_evidence_precision": len(gold & used) / len(used) if used else None,
                                "fact_review": "pending"}
                        except Exception as exc:
                            record = {"case_id": case.case_id, "conversation_id": group, "turn": case.turn,
                                      "status": "failed", "error_type": type(exc).__name__}
                            if isinstance(getattr(exc, "code", None), str):
                                record["error_code"] = exc.code
                            if getattr(exc, "generation", None):
                                record["generation"] = agent.operations.gateway.sanitize(session, exc.generation.to_dict(secrets))
                        records.append(record)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        summarize("running")
                        if progress:
                            progress(f"{len(records)}/{sum(map(len, groups.values()))} {case.case_id}: {record['status']}")
                        code = record.get("error_code", "")
                        service_failure = code.startswith("http_") or code == "connection_failed"
                        consecutive_service_failures = consecutive_service_failures + 1 if service_failure else 0
                        if code in {"http_400", "http_401", "http_403", "http_404"} or consecutive_service_failures >= 3:
                            return summarize("halted_service_failure")
                finally:
                    agent.clear_session(session)
    except BaseException:
        summarize("interrupted")
        raise
    summary = summarize("finished")
    (output_dir / "report.md").write_text(
        f"# Phase 4 conversation replay\n\nMode: {mode}.\n\n"
        f"Processed {summary['processed']}/{summary['turns']} turns across {len(groups)} conversations. "
        f"Decision matches: {summary['decision_matches']}; tool-name matches: {summary['tool_call_matches']}; "
        f"scenarios passing both routing checks: {summary['scenarios_passing_routing']}.\n\n"
        "These checks do not measure semantic answer quality. Gemini generation and grading remain deferred "
        "until project completion. Offline policy fallback can return original English source extracts.\n", encoding="utf-8")
    return summary
