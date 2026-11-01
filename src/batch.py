"""Ordered audio-folder runner with immutable outputs and resumable context."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .evaluation.runner import save_json
from .speech.audio import EXTENSIONS
from .speech.engine import VoiceTurn
from .schemas import AgentResponse, Decision


@dataclass(frozen=True)
class AudioCase:
    path: Path
    conversation_id: str
    turn: int
    language: str


def audio_cases(input_dir: Path, *, manifest: Path | None = None, language="en-IN"):
    root = input_dir.resolve()
    if not root.is_dir():
        raise ValueError("Input directory does not exist")
    if manifest:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Audio manifest must be a JSON array")
        cases = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"conversation_id", "turn", "file", "language"}:
                raise ValueError("Manifest rows need conversation_id, turn, file and language")
            if not isinstance(row["file"], str) or not isinstance(row["conversation_id"], str) or not row["conversation_id"].strip():
                raise ValueError("Manifest IDs and file paths must be nonempty strings")
            if type(row["turn"]) is not int or row["turn"] < 1:
                raise ValueError("Manifest turn must be a positive integer")
            path = (root / row["file"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Manifest audio must be a file inside the input directory")
            cases.append(AudioCase(path, row["conversation_id"], row["turn"], row["language"]))
    else:
        cases = [AudioCase(p.resolve(), p.name, 1, language) for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if not cases:
        raise ValueError("No supported audio files found")
    from .speech.provider import LANGUAGES
    seen = set()
    for case in cases:
        if not case.path.is_relative_to(root) or case.language not in LANGUAGES:
            raise ValueError("Unsupported language or audio outside the input directory")
        if (case.conversation_id, case.turn) in seen:
            raise ValueError("Duplicate conversation turn")
        seen.add((case.conversation_id, case.turn))
    cases.sort(key=lambda c: (c.conversation_id, c.turn))
    for group in {c.conversation_id for c in cases}:
        turns = [c.turn for c in cases if c.conversation_id == group]
        if turns != list(range(1, len(turns)+1)):
            raise ValueError("Conversation turns must be consecutive, starting at one")
    return tuple(cases)


def run_batch(engine, input_dir: Path, output_dir: Path, *, manifest=None, language="en-IN", resume=False, progress=None):
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir.is_relative_to(engine.agent.operations.dataset_dir):
        raise ValueError('Output must be outside the supplied dataset')
    if output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise ValueError("Input and output directories must be separate")
    cases = audio_cases(input_dir, manifest=manifest, language=language)
    from .speech.audio import MAX_BYTES
    fingerprints = [{"audio_sha256": hashlib.sha256(c.path.read_bytes()).hexdigest() if c.path.stat().st_size<=MAX_BYTES else None,
                     "size_bytes":c.path.stat().st_size,"conversation_id": c.conversation_id,
                     "turn": c.turn, "language": c.language} for c in cases]
    from .retrieval.index import source_fingerprint
    dataset_dir = engine.agent.operations.dataset_dir
    metadata = {"cases": fingerprints, "stt_model": engine.speech.settings.gnani_stt_model,
                "tts_model": engine.speech.settings.gnani_tts_model, "tts_voice": engine.speech.settings.gnani_tts_voice,
                "agent_mode": "live" if engine.agent.rag else "offline",
                "llm_model": engine.speech.settings.llm_model,
                "generation_parameters": engine.speech.settings.generation_parameters(),
                "dataset_sha256": {str(p.relative_to(dataset_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(dataset_dir.rglob('*')) if p.suffix in {'.csv','.md','.py'} and p.name != 'tickets_created.csv'},
                "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).resolve().parent.rglob('*')) if p.suffix in {'.py','.txt'}}}
    path = output_dir / "manifest.json"
    if resume:
        old = json.loads(path.read_text())
        if old["configuration"] != metadata:
            raise ValueError("Resume inputs/settings differ from the saved run")
        run_id = old["run_id"]
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        run_id = uuid4().hex
        save_json(path, {"run_id": run_id, "configuration": metadata, "resume_note": "Partial conversations replay prior input audio for state; successful tickets remain idempotent. No proof transcript is stored."})
    records = []
    groups = sorted({c.conversation_id for c in cases})
    for group in groups:
        group_cases = [(i,c) for i,c in enumerate(cases) if c.conversation_id == group]
        session = run_id + ":" + hashlib.sha256(group.encode()).hexdigest()[:20]
        engine.agent.clear_session(session)
        previous = []
        for i, case in group_cases:
            directory = output_dir / f"clip-{i+1:04d}"
            result_path = directory / "result.json"
            previous.append(json.loads(result_path.read_text()) if result_path.exists() else None)
        if resume and all(r and r["status"] in {"completed", "needs_repeat"} and (output_dir / f"clip-{i+1:04d}" / "reply.wav").is_file() for (i,_),r in zip(group_cases,previous)):
            records.extend(previous)
            continue
        try:
            for (i, case), old in zip(group_cases, previous):
                directory = output_dir / f"clip-{i+1:04d}"
                directory.mkdir(exist_ok=True)
                result_path = directory / "result.json"
                # If this is a partial conversation, replay earlier external audio
                # to reconstruct identity without storing verification secrets.
                if resume and old and old["status"] == "completed" and (directory / "reply.wav").is_file():
                    engine.process(case.path, session, case.language)
                    records.append(old)
                    continue
                def checkpoint(result):
                    data = engine.agent.operations.gateway.sanitize(session, result.to_dict())
                    save_json(directory / "text_checkpoint.json", data)
                saved = old or (json.loads((directory/'text_checkpoint.json').read_text()) if (directory/'text_checkpoint.json').exists() else None)
                last_in_group = case.turn == len(group_cases)
                if resume and last_in_group and saved and saved.get('response') and (saved.get('status')=='text_ready' or saved.get('error',{} ) and saved['error'].get('stage')=='tts'):
                    from .speech.engine import restored_turn
                    result = engine.synthesize(restored_turn(saved))
                else:
                    result = engine.process(case.path, session, case.language, on_text=checkpoint)
                data = engine.agent.operations.gateway.sanitize(session, result.to_dict())
                data["clip_id"] = f"clip-{i+1:04d}"
                data["input_sha256"] = fingerprints[i]["audio_sha256"]
                if result.response:
                    (directory / "reply.txt").write_text(data["response"]["reply"] + "\n", encoding="utf-8")
                    save_json(directory / "evidence.json", {"evidence": data["response"]["evidence"], "evidence_ids": data["response"]["evidence_ids"]})
                if result.audio:
                    (directory / "reply.wav").write_bytes(result.audio)
                elif (directory / "reply.wav").exists():
                    (directory / "reply.wav").unlink()
                save_json(result_path, data)
                records.append(data)
                if progress:
                    progress(f"{data['clip_id']}: {data['status']}")
        finally:
            engine.agent.clear_session(session)
    complete = sum(r.get("voice_complete", False) for r in records)
    summary = {"clips": len(cases), "voice_completed": complete, "needs_repeat": sum(r["status"] == "needs_repeat" for r in records),
               "failed": sum(r["status"] == "failed" for r in records), "status": "completed" if complete == len(cases) else "partial_failure"}
    save_json(output_dir / "summary.json", summary)
    return summary
