"""VoiceOps setup checks, prompt-only answers and reproducible baseline runs."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
from uuid import uuid4

from .config import ConfigurationError, Settings
from .dataset import DatasetError, inventory, validate_supplied_dataset
from .evaluation.cases import load_conversations, load_questions
from .integrations import IntegrationError, gnani_smoke, llm_smoke
from .llm import ChatCompletionClient, LLMError
from .agent import AgentOutputError, BaselineAgent
from .schemas import CustomerInput
from .evaluation.experiment import baseline, sha256, write_report
from .evaluation.judge import grade_run
from .evaluation.runner import save_json


def practice_inventory(root: Path) -> dict:
    questions = load_questions(root)
    groups = load_conversations(root)
    return {"questions": len(questions), "conversations": len(groups),
            "conversation_turns": sum(len(cases) for cases in groups.values()),
            "question_languages": sorted({case.language for case in questions}),
            "agent_runs": 0, "scores": "not_run"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Override DATASET_DIR")
    parser.add_argument("--env-file", type=Path, help="Load an explicit environment file instead of .env")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Offline configuration, data and SDK readiness")
    check.add_argument("--validate", action="store_true", help="Also run the supplied dataset validator")
    sub.add_parser("eval-inventory", help="Load practice cases; does not run an agent or report scores")
    smoke = sub.add_parser("smoke", help="Live API calls; may incur provider charges")
    smoke.add_argument("--component", choices=("all", "llm", "gnani"), default="all")
    smoke.add_argument("--audio", type=Path, help="Optional own recording for STT; otherwise transcribe synthesized test speech")
    ask = sub.add_parser("ask", help="One new prompt-only support conversation (live API call)")
    ask.add_argument("message")
    ask.add_argument("--stage", choices=("baseline", "rag"), default="baseline")
    run = sub.add_parser("baseline", help="Run actual Step 1 practice answers (live API calls)")
    run.add_argument("--input", choices=("original", "asr", "both"), default="original")
    run.add_argument("--limit", type=int, help="Optional small integration run, not the full baseline")
    run.add_argument("--judge", action="store_true", help="Make separate semantic-grading API calls after generation")
    run.add_argument("--stage", choices=("baseline", "rag"), default="baseline")
    grade = sub.add_parser("grade", help="Grade a finished run without regenerating its answers")
    grade.add_argument("run_dir", type=Path)
    index_command = sub.add_parser("index", help="Build the local multilingual vector index; no Gemini calls")
    index_command.add_argument("--download-model", action="store_true", help="Permit public embedding-model download")
    retrieve = sub.add_parser("retrieve", help="Search the local index without generating an answer")
    retrieve.add_argument("query")
    retrieve.add_argument("--k", type=int, default=12)
    retrieve.add_argument("--delivered-on", type=str, help="Known delivery date for historical-policy applicability")
    retrieval_eval = sub.add_parser("eval-retrieval", help="Evaluate retrieval locally without Gemini")
    retrieval_eval.add_argument("--input", choices=("original", "asr", "both"), default="both")
    retrieval_eval.add_argument("--k", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        settings = Settings.load(env_file=args.env_file)
        dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else settings.dataset_dir
        def local_index(*, build=False, download=False):
            from .retrieval.embeddings import LocalEmbedder
            from .retrieval.index import KnowledgeIndex
            if settings.runtime_dir.is_relative_to(dataset_dir):
                raise ConfigurationError("RUNTIME_DIR must be outside the dataset directory")
            embedder = LocalEmbedder(settings.runtime_dir / "models", allow_download=download)
            path = settings.runtime_dir / "retrieval"
            if build:
                result = KnowledgeIndex.build(dataset_dir, embedder)
                result.save(path)
                return result
            return KnowledgeIndex.load(path, embedder, dataset_dir)
        if args.command == "eval-inventory":
            result = practice_inventory(dataset_dir)
        elif args.command == "index":
            index = local_index(build=True, download=args.download_model)
            result = {"status": "built", "chunks": len(index.evidence), "embedding_model": index.embedder.name,
                      "fingerprint": index.fingerprint, "gemini_calls": 0}
        elif args.command == "retrieve":
            from datetime import date
            from dataclasses import asdict
            index = local_index()
            found = index.search(args.query, k=args.k,
                                  delivered_on=date.fromisoformat(args.delivered_on) if args.delivered_on else None)
            result = {"query": found.query, "candidates": [{"evidence_id": h.evidence.evidence_id,
                       "similarity": h.similarity, "score": h.score} for h in found.candidates],
                      "context": [asdict(e) for e in found.context], "gemini_calls": 0}
        elif args.command == "eval-retrieval":
            from .evaluation.retrieval import evaluate_retrieval
            index = local_index()
            reports = []
            for variant in (("original", "asr") if args.input == "both" else (args.input,)):
                output = settings.runtime_dir / "retrieval_evaluations" / (variant + "-" + uuid4().hex[:12])
                reports.append({"output_dir": str(output), **evaluate_retrieval(index, dataset_dir, output,
                                                                                k=args.k, asr=variant == "asr")})
            result = {"reports": reports}
        elif args.command == "check":
            if args.validate:
                validate_supplied_dataset(dataset_dir)
            dependencies = {}
            for name in ("gnani-vachana", "python-dotenv"):
                try:
                    dependencies[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    dependencies[name] = "not_installed"
            result = {"status": "offline_passed", "configuration": settings.readiness(),
                      "dataset_dir": str(dataset_dir), "dataset": inventory(dataset_dir),
                      "practice": practice_inventory(dataset_dir), "dependencies": dependencies,
                      "supplied_validator": "passed" if args.validate else "not_run",
                      "live_connectivity": "not_tested"}
        elif args.command == "ask":
            client = ChatCompletionClient(settings)
            agent = BaselineAgent(client)
            if args.stage == "rag":
                from .rag import RagAgent
                agent = RagAgent(client, local_index())
            response = agent.respond(CustomerInput(args.message, uuid4().hex))
            result = response.to_dict((settings.llm_api_key, settings.gnani_api_key))
        elif args.command == "baseline":
            runs = []
            for variant in (("original", "asr") if args.input == "both" else (args.input,)):
                path, summary = baseline(settings, dataset_dir=dataset_dir, variant=variant,
                                         limit=args.limit, judge=args.judge,
                                         stage=args.stage, index=local_index() if args.stage == "rag" else None,
                                         progress=lambda message: print(message, file=sys.stderr, flush=True))
                runs.append({"output_dir": str(path), "input_variant": variant, **summary})
                grading_path = path / "grading_summary.json"
                if grading_path.is_file():
                    runs[-1]["grading"] = json.loads(grading_path.read_text())
                if summary["run_status"] != "finished":
                    break
            print(json.dumps({"runs": runs}, indent=2))
            return 0 if all(run["failed"] == 0 and run.get("grading", {}).get("status", "complete") == "complete" for run in runs) else 2
        elif args.command == "grade":
            path = args.run_dir.resolve()
            if path.is_relative_to(dataset_dir):
                raise ConfigurationError("Evaluation output must be outside the dataset directory")
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest.get("practice_sha256") != sha256(dataset_dir / "practice/questions.csv"):
                raise ValueError("Practice data differs from the saved run")
            cases = load_questions(dataset_dir, asr=manifest["input_variant"] == "asr")
            result = grade_run(path, cases, ChatCompletionClient(settings),
                               secrets=(settings.llm_api_key, settings.gnani_api_key),
                               progress=lambda message: print(message, file=sys.stderr, flush=True))
            write_report(path)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 2
        else:
            if settings.runtime_dir.is_relative_to(dataset_dir):
                raise ConfigurationError("RUNTIME_DIR must be outside the dataset directory")
            run_dir = settings.runtime_dir / "smoke" / uuid4().hex
            checks = {}
            for component in (("llm", "gnani") if args.component == "all" else (args.component,)):
                if component == "llm" and not (settings.llm_endpoint and settings.llm_model):
                    checks[component] = {"status": "pending", "reason": "Set LLM_ENDPOINT, LLM_MODEL and provider credentials if required"}
                    continue
                if component == "gnani" and not settings.gnani_api_key:
                    checks[component] = {"status": "pending", "reason": "Set GNANI_API_KEY"}
                    continue
                try:
                    checks[component] = (llm_smoke(settings) if component == "llm"
                                         else gnani_smoke(settings, run_dir, args.audio))
                except (IntegrationError, ImportError) as exc:
                    checks[component] = {"status": "failed", "reason": str(exc) if isinstance(exc, IntegrationError) else "Install requirements.txt"}
            result = {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": checks}
            run_dir.mkdir(parents=True, exist_ok=True)
            save_json(run_dir / "checks.json", {**result, "configuration": settings.readiness()})
            result["report_path"] = str(run_dir / "checks.json")
            if any(item["status"] != "passed" for item in checks.values()):
                print(json.dumps(result, indent=2))
                return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ConfigurationError, DatasetError, OSError, ValueError, LLMError) as exc:
        # Configuration/dataset errors contain paths/field names, not private rows.
        print(json.dumps({"status": "failed", "reason": str(exc)}), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "reason": "Partial run outputs were preserved"}), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
