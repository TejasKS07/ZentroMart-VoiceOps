"""Separate customer inputs from labels to avoid evaluation-answer leakage."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..dataset import DatasetError, indexed, read_csv
from ..schemas import CustomerInput


def separated(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(" ; ") if part.strip())


@dataclass(frozen=True, repr=False)
class EvaluationCase:
    case_id: str
    customer: CustomerInput
    language: str
    turn: int
    acceptable_decisions: tuple[str, ...]
    gold_evidence: tuple[str, ...]
    labels: dict[str, str] = field(repr=False)


def load_questions(root: Path, *, asr: bool = False) -> tuple[EvaluationCase, ...]:
    rows = read_csv(root / "practice/questions.csv", (
        "id", "question", "asr_text", "language", "acceptable_decisions",
        "gold_evidence", "must_include", "must_not_include"))
    indexed(rows, ("id",), "practice questions")
    return tuple(EvaluationCase(
        row["id"], CustomerInput(row["asr_text" if asr else "question"], row["id"]),
        row["language"], 1, separated(row["acceptable_decisions"]),
        separated(row["gold_evidence"]), dict(row),
    ) for row in rows)


def load_conversations(root: Path) -> dict[str, tuple[EvaluationCase, ...]]:
    rows = read_csv(root / "practice/conversations.csv", (
        "conversation_id", "turn", "customer_message", "language", "expected_decision",
        "expected_tool_calls", "gold_evidence"))
    indexed(rows, ("conversation_id", "turn"), "practice turns")
    groups = defaultdict(list)
    for row in rows:
        try:
            turn = int(row["turn"])
        except ValueError:
            raise DatasetError("Invalid conversation turn number") from None
        session = row["conversation_id"]
        groups[session].append(EvaluationCase(
            f"{session}:{turn}", CustomerInput(row["customer_message"], session),
            row["language"], turn, (row["expected_decision"],),
            separated(row["gold_evidence"]), dict(row),
        ))
    result = {}
    for session, cases in sorted(groups.items()):
        ordered = sorted(cases, key=lambda case: case.turn)
        if [case.turn for case in ordered] != list(range(1, len(ordered) + 1)):
            raise DatasetError(f"Nonconsecutive conversation turns: {session}")
        result[session] = tuple(ordered)
    return result

