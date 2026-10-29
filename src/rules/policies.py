"""Read current policy values from KB sections, never from practice answers."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import re

from ..dataset import PublicKnowledge


def number(pattern: str, text: str, group: int = 1) -> int:
    match = re.search(pattern, text, re.I)
    if not match:
        raise ValueError("Policy format changed; required rule could not be parsed")
    return int(match.group(group).replace(",", ""))


def table(text: str) -> list[list[str]]:
    return [[cell.strip().strip("`") for cell in line.split("|")[1:-1]]
            for line in text.splitlines() if line.startswith("|") and "---" not in line][1:]


@dataclass(frozen=True)
class RefundWindow:
    minimum: int
    maximum: int
    unit: str


@dataclass(frozen=True)
class PolicyRules:
    return_windows: dict[str, int | None]
    refund_windows: dict[str, RefundWindow]
    electronics_cutover: date
    old_electronics_days: int
    festive_start: date
    festive_end: date
    festive_days: int
    pickup_fee: Decimal
    damage_hours: int
    high_value_threshold: Decimal
    refund_initiation_days: int
    overdue_escalation_days: int
    prior_contact_threshold: int
    holidays: frozenset[date]
    sections: dict[str, str]

    @classmethod
    def from_knowledge(cls, knowledge: PublicKnowledge) -> "PolicyRules":
        sections = {e.evidence_id: e.text for d in knowledge.documents if d.metadata["status"] == "current" for e in d.sections}
        windows = {}
        for _, window, category in table(sections["RET-2"]):
            windows[category] = number(r"(\d+)", window) if re.search(r"\d", window) else None
        refund_windows = {}
        method_names = {"UPI": "upi", "Credit or debit card": "card", "Net banking": "netbanking",
                        "ZentroMart Wallet": "wallet", "Bank transfer (NEFT) for cash-on-delivery orders": "cod_bank",
                        "ZentroMart Wallet for cash-on-delivery orders": "cod_wallet"}
        for label, timeline in table(sections["REF-2"]):
            if label not in method_names:
                raise ValueError("Unknown refund payment method in policy")
            if "hours" in timeline:
                refund_windows[method_names[label]] = RefundWindow(0, number(r"(\d+)\s*hours", timeline), "hours")
            else:
                match = re.search(r"(\d+)\s+to\s+(\d+)\s+business days", timeline)
                if not match:
                    raise ValueError("Unrecognized refund timeline")
                refund_windows[method_names[label]] = RefundWindow(int(match[1]), int(match[2]), "business_days")
        change = re.search(r"From\s+\*\*(\d{1,2} \w+ \d{4})\*\*", sections["RET-3"])
        festive = re.search(r"between (\d{2} \w+) and (\d{2} \w+ \d{4})", sections["RET-8"])
        if not change or not festive:
            raise ValueError("Return policy date rule could not be parsed")
        end = datetime.strptime(festive[2], "%d %B %Y").date()
        return cls(windows, refund_windows, datetime.strptime(change[1], "%d %B %Y").date(),
                   number(r"earlier (\d+)-day", sections["RET-3"]),
                   datetime.strptime(festive[1] + f" {end.year}", "%d %B %Y").date(), end,
                   number(r"within (\d+) days", sections["RET-8"]),
                   Decimal(number(r"₹([\d,]+)", sections["RET-6"])),
                   number(r"(\d+) hours", sections["DWM-1"]),
                   Decimal(number(r"₹([\d,]+)", sections["DWM-3"])),
                   number(r"within (\d+) business days", sections["RET-7"]),
                   number(r"more than (\d+) business days", sections["ESC-1"]),
                   number(r"when it is (\d+) or more", sections["ESC-1"]),
                   frozenset(date.fromisoformat(row["date"]) for row in knowledge.holidays), sections)
