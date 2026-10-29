"""Public lookups and calculations on verified order facts; never execute refunds."""

from datetime import date
from pathlib import Path

from ..dataset import PublicKnowledge
from ..rules.calculations import Calculation, PolicyCalculator
from ..rules.policies import PolicyRules
from ..schemas import Evidence
from .gateway import ToolGateway, identifier


class SupportOperations:
    def __init__(self, dataset_dir: Path, runtime_dir: Path, *, gateway: ToolGateway | None = None):
        self.knowledge = PublicKnowledge.load(dataset_dir)
        self.calculator = PolicyCalculator(PolicyRules.from_knowledge(self.knowledge))
        self.gateway = gateway or ToolGateway(dataset_dir, runtime_dir)

    def product(self, sku: str) -> Calculation:
        sku = sku.strip().upper()
        row = self.knowledge.products.get(sku)
        evidence = Evidence(f"products.csv:{sku}", "data/products.csv", "\n".join(f"{key}: {value}" for key, value in row.items()),
                            {"doc_type": "catalog", "status": "current"}) if row else None
        return Calculation({"found": row is not None, "product": dict(row) if row else None},
                           (evidence.evidence_id,) if evidence else (), (evidence,) if evidence else ())

    def pincode(self, pincode: str) -> Calculation:
        pincode = identifier(pincode, (6,))
        row = self.knowledge.pincodes.get(pincode)
        # An absent pincode is not evidence of nonserviceability.
        evidence = Evidence(f"pincodes.csv:{pincode}", "data/pincodes.csv", "\n".join(f"{key}: {value}" for key, value in row.items()),
                            {"doc_type": "reference", "status": "current"}) if row else None
        return Calculation({"found": row is not None, "availability": dict(row) if row else None},
                           ("DEL-5", evidence.evidence_id) if evidence else ("DEL-5",), (evidence,) if evidence else ())

    def return_eligibility(self, session_id: str, line_no: str) -> Calculation:
        facts = self.gateway.order_facts(session_id)
        item = next((i for i in facts["items"] if str(i["line_no"]) == str(line_no)), None)
        if not item:
            raise ValueError("Item line not found in verified order")
        if not facts.get("delivered_on"):
            return Calculation({"needs_delivery": True, "within_window": None}, ("RET-2",))
        return self.calculator.return_window(item["return_category"], date.fromisoformat(facts["delivered_on"]),
                                             date.fromisoformat(facts["placed_at"][:10]))

    def refund_status(self, session_id: str, kind: str) -> Calculation:
        if kind not in {"return", "cancellation", "rto"}:
            raise ValueError("Select return, cancellation or rto refund")
        facts = self.gateway.order_facts(session_id)
        prefix = kind + "_refund_"
        initiated = facts.get(prefix + "initiated_on")
        values = {"kind": kind, "initiated": bool(initiated), "amount_inr": facts.get(prefix + "amount_inr"),
                  "rrn": facts.get(kind + "_rrn"), "completed_on": facts.get(prefix + "completed_on")}
        method = facts.get(prefix + "method")
        if initiated and method:
            timeline = self.calculator.refund_timeline(method, date.fromisoformat(initiated))
            values.update(timeline.values)
        else:
            values["needs_initiation_or_method"] = True
        return Calculation(values, ("REF-1", "REF-2", "REF-6", "KB-00.3"))

    def refund_quote(self, session_id: str, quantities: dict[str, int], *, reason: str,
                     cancellation: bool = False) -> Calculation:
        facts = self.gateway.order_facts(session_id)
        return self.calculator.refund_quote(facts, quantities, is_plus=facts["is_plus"],
                                            reason=reason, cancellation=cancellation)
