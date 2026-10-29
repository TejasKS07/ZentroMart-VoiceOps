"""Baseline orchestration with immutable experiment inputs and a short report."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from uuid import uuid4

from ..agent import BASELINE_PROMPT, BASELINE_PROMPT_SHA256, BASELINE_PROMPT_VERSION, BaselineAgent
from ..config import PROJECT_ROOT, Settings
from ..llm import ChatCompletionClient
from .cases import load_questions
from .judge import grade_run
from .runner import run_questions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    rate = summary["decision_match_rate"]
    is_rag = manifest.get("stage") == "step2_rag"
    lines = ["# Step 2 grounded agent" if is_rag else "# Step 1 prompt-only baseline", f"Run: `{run_dir.name}`",
             f"Model: `{manifest['requested_model']}`; input variant: `{manifest['input_variant']}`.",
             f"Processed: {summary['processed']}/{summary['cases']}; completed: {summary['completed']}; failed: {summary['failed']}.",
             "The agent received policy/catalog retrieval chunks and the customer message. No private orders, reference answers or gold evidence were supplied." if is_rag else "The agent received only its role/limits prompt and customer message. No KB, catalog, order data, tools, reference answers or gold evidence were supplied.",
             f"Decision match rate: {rate:.1%}. Decision matches alone are not factual accuracy." if rate is not None else "Decision match rate: unavailable."]
    grading_path = run_dir / "grading_summary.json"
    if grading_path.is_file():
        grading = json.loads(grading_path.read_text())
        lines += [f"Provisional model-judged correct: {grading['correct']}/{grading['cases']}; unresolved: {grading['unresolved']}.",
                  f"Provisional accuracy: {grading['accuracy']:.1%}." if grading['accuracy'] is not None else "Provisional accuracy: pending unresolved grades.", grading["note"]]
    else:
        lines += ["Factual correctness has not been graded; do not report factual accuracy yet."]
    lines += ["Preserve this run for Step 2. Keep the model, generation settings, selected cases, input variant, and grading method fixed; add retrieval as the experimental change.",
              "Raw model text is retained in generation traces; answers are not repaired or manually edited. See manifest.json, results.jsonl, summary.json and optional judge files."]
    (run_dir / "report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def baseline(settings: Settings, *, dataset_dir: Path, variant: str = "original", limit: int | None = None,
             judge: bool = False, progress=None, stage: str = "baseline", index=None) -> tuple[Path, dict]:
    client = ChatCompletionClient(settings)
    if stage not in {"baseline", "rag"} or (stage == "rag" and index is None):
        raise ValueError("RAG experiments need a loaded knowledge index")
    if variant not in {"original", "asr"}:
        raise ValueError("Unknown input variant")
    if settings.runtime_dir.is_relative_to(dataset_dir):
        raise ValueError("RUNTIME_DIR must be outside the dataset directory")
    cases = load_questions(dataset_dir, asr=variant == "asr")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[:limit]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    run_dir = settings.runtime_dir / stage / run_id
    code_files = sorted(p for p in (PROJECT_ROOT / "src").rglob("*") if p.suffix in {".py", ".txt"})
    metadata = {"stage": "step1_prompt_only", "input_variant": variant, "requested_model": settings.llm_model,
                "endpoint": settings.llm_endpoint, "generation_parameters": settings.generation_parameters(),
                "omitted_generation_parameters": "Provider defaults apply; keep provider settings fixed between stages.",
                "timeout_seconds": settings.timeout_seconds, "transport_max_attempts": client.max_attempts,
                "reference_date": settings.reference_date.isoformat(), "python_version": platform.python_version(),
                "dependencies": {n: importlib.metadata.version(n) for n in ("requests", "gnani-vachana", "python-dotenv")},
                "prompt_version": BASELINE_PROMPT_VERSION, "system_prompt": BASELINE_PROMPT,
                "prompt_sha256": BASELINE_PROMPT_SHA256,
                "practice_sha256": sha256(dataset_dir / "practice/questions.csv"),
                "source_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code_files},
                "full_practice_run": len(cases) == len(load_questions(dataset_dir))}
    secrets = (settings.llm_api_key, settings.gnani_api_key)
    agent = BaselineAgent(client)
    if stage == "rag":
        from ..rag import RAG_PROMPT, RAG_PROMPT_SHA256, RAG_PROMPT_VERSION, RagAgent
        agent = RagAgent(client, index)
        metadata.update(stage="step2_rag", system_prompt=RAG_PROMPT, prompt_sha256=RAG_PROMPT_SHA256,
                        prompt_version=RAG_PROMPT_VERSION, index_fingerprint=index.fingerprint,
                        embedding_model=index.embedder.name, top_k=agent.top_k)
    summary = run_questions(cases, agent.respond, run_dir,
                            secrets=secrets, metadata=metadata, progress=progress)
    write_report(run_dir)
    if judge and summary["run_status"] == "finished":
        grade_run(run_dir, cases, client, secrets=secrets, progress=progress)
        write_report(run_dir)
    return run_dir, summary
