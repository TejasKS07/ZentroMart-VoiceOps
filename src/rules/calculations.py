"""Calendar, eligibility, settlement and refund estimates; no order mutations."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from ..config import REFERENCE_DATE
from ..schemas import Evidence
from .policies import PolicyRules


@dataclass(frozen=True)
class Calculation:
    values: dict[str, Any]
    evidence_ids: tuple[str, ...]
    source_evidence: tuple[Evidence, ...] = ()


class PolicyCalculator:
    def __init__(self, rules: PolicyRules, today: date = REFERENCE_DATE):
        self.rules = rules
        self.today = today

    def is_business_day(self, day: date) -> bool:
        return day.weekday() != 6 and day not in self.rules.holidays

    def add_business_days(self, start: date, days: int) -> date:
        if type(days) is not int or days < 0:
            raise ValueError("Business-day offset must be a nonnegative integer")
        current = start
        while days:
            current += timedelta(days=1)
            if self.is_business_day(current):
                days -= 1
        return current

    def business_days_elapsed(self, start: date, end: date) -> int:
        count = 0
        while start < end:
            start += timedelta(days=1)
            count += self.is_business_day(start)
        return count

    def return_window(self, category: str, delivered_on: date, placed_on: date) -> Calculation:
        if delivered_on > self.today or placed_on > delivered_on:
            raise ValueError("Invalid order delivery/placement dates")
        if category not in self.rules.return_windows:
            raise ValueError("Unknown catalog return category")
        days = self.rules.return_windows[category]
        evidence = ["RET-2", "KB-00.2"]
        if category == "electronics" and delivered_on < self.rules.electronics_cutover:
            days = self.rules.old_electronics_days
            evidence.append("RET-3")
        if category == "fashion" and self.rules.festive_start <= placed_on <= self.rules.festive_end:
            days = self.rules.festive_days
            evidence.append("RET-8")
        deadline = delivered_on + timedelta(days=days) if days is not None else None
        return Calculation({"category": category, "window_days": days,
                            "deadline": deadline.isoformat() if deadline else None,
                            "within_window": deadline is not None and self.today <= deadline,
                            "conditions_must_be_checked": True,
                            "not_returnable_for_change_of_mind": days is None}, tuple(evidence))

    def refund_timeline(self, method: str, initiated_on: date | None) -> Calculation:
        if method not in self.rules.refund_windows:
            raise ValueError("Unknown refund method; COD needs wallet/bank choice")
        window = self.rules.refund_windows[method]
        values = {"method": method, "minimum": window.minimum, "maximum": window.maximum,
                  "unit": window.unit, "initiated_on": initiated_on.isoformat() if initiated_on else None}
        if initiated_on and initiated_on > self.today:
            raise ValueError("Refund initiation cannot be in the future")
        if initiated_on and window.unit == "business_days":
            earliest = self.add_business_days(initiated_on, window.minimum)
            latest = self.add_business_days(initiated_on, window.maximum)
            overdue = self.business_days_elapsed(latest, self.today)
            values.update(earliest_date=earliest.isoformat(), latest_date=latest.isoformat(),
                          overdue_business_days=overdue,
                          requires_escalation=overdue > self.rules.overdue_escalation_days)
        elif initiated_on:
            # Date-only records cannot establish an exact two-hour deadline.
            values.update(exact_deadline=None, needs_initiation_time_for_exact_deadline=True)
        return Calculation(values, ("REF-2", "KB-00.3", "ESC-1"))

    def refund_initiation(self, pickup_on: date | None, *, is_plus: bool,
                          qc_passed: bool) -> Calculation:
        if pickup_on is None:
            return Calculation({"ready": False, "needs_pickup": True}, ("RET-7",))
        if is_plus:
            return Calculation({"initiation_date": pickup_on.isoformat(), "ready": True}, ("RET-7", "REF-1"))
        return Calculation({"ready": qc_passed,
                            "latest_initiation_date": self.add_business_days(pickup_on, self.rules.refund_initiation_days).isoformat(),
                            "requires_qc_pass": True}, ("RET-7", "REF-1", "KB-00.3"))

    def cancellation(self, *, status: str, grocery: bool = False,
                     slot_start: datetime | None = None, at: datetime | None = None) -> Calculation:
        unshipped = {"placed", "confirmed", "processing", "packed"}
        statuses = unshipped | {"shipped", "out_for_delivery", "delivered", "cancelled", "returned_to_origin",
                                "delivery_attempted", "return_requested", "return_picked_up", "return_refund_initiated", "returned_refunded"}
        if status not in statuses:
            raise ValueError("Unknown order status")
        if grocery and status in unshipped:
            if slot_start is None or at is None:
                return Calculation({"needs_clarification": True,
                                    "missing": "slot_start" if slot_start is None else "current_time"}, ("CAN-3",))
            if (slot_start.tzinfo is None) != (at.tzinfo is None):
                raise ValueError("Cancellation timestamps must use consistent timezones")
            from .policies import number
            hours = number(r"up to (\d+) hours", self.rules.sections["CAN-3"])
            cutoff = slot_start - timedelta(hours=hours)
            return Calculation({"can_customer_cancel": at <= cutoff,
                                "cutoff": cutoff.isoformat(), "assistant_can_cancel": False}, ("CAN-3",))
        before_shipping = status in unshipped
        return Calculation({"can_customer_cancel": before_shipping, "assistant_can_cancel": False,
                            "customer_action": "My Orders" if before_shipping else "review_refusal_or_return_options" if status in {"shipped", "out_for_delivery", "delivery_attempted"} else "cancellation_not_applicable"},
                           ("CAN-1",) if before_shipping else ("CAN-2", "REF-1"))

    def refund_quote(self, order: dict, quantities: dict[str, int], *, is_plus: bool,
                     reason: str, cancellation: bool = False) -> Calculation:
        if reason not in {"change_of_mind", "damaged", "defective", "wrong"}:
            raise ValueError("Unknown return reason")
        if not quantities:
            raise ValueError("Select at least one item line")
        if any(not isinstance(line, str) for line in quantities):
            raise ValueError("Item line identifiers must be strings")
        items = {str(item["line_no"]): item for item in order["items"]}
        gross = Decimal(0)
        selected_categories = set()
        for line, qty in quantities.items():
            item = items.get(str(line))
            if item is None or type(qty) is not int or not 0 < qty <= int(item["qty"]):
                raise ValueError("Invalid item line or return quantity")
            gross += Decimal(str(item["unit_price_inr"])) * qty
            selected_categories.add(item["return_category"])
        subtotal = Decimal(str(order["subtotal_inr"]))
        if subtotal <= 0 or gross > subtotal:
            raise ValueError("Invalid order subtotal")
        discount = Decimal(str(order.get("coupon_discount_inr") or 0))
        if not 0 <= discount <= subtotal:
            raise ValueError("Invalid coupon discount")
        remaining = subtotal - gross
        allocation = discount * gross / subtotal
        minimum = Decimal(str(order.get("coupon_min_order_inr") or 0))
        lost_discount = discount-allocation if discount and remaining and remaining < minimum else Decimal(0)
        fault = reason in {"damaged", "defective", "wrong"}
        pickup = self.rules.pickup_fee if not cancellation and "fashion" in selected_categories and not is_plus and not fault else Decimal(0)
        fees = Decimal(0)
        if remaining == 0 and (fault or cancellation):
            fees = Decimal(str(order.get("delivery_fee_inr") or 0)) + Decimal(str(order.get("cod_fee_inr") or 0))
        if cancellation and order["status"] not in {"placed", "confirmed", "processing", "packed"}:
            raise ValueError("Cancellation refund quote requires an unshipped order")
        amount = max(Decimal(0), gross-allocation-lost_discount-pickup+fees).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return Calculation({"estimate_inr": int(amount), "item_gross_inr": str(gross),
                            "allocated_coupon_inr": str(allocation), "lost_remaining_discount_inr": str(lost_discount),
                            "reverse_pickup_fee_inr": str(pickup), "refundable_delivery_cod_fees_inr": str(fees),
                            "estimate_only": True, "refund_issued": False}, ("REF-4", "REF-5", "RET-6"))

    def escalation_reasons(self, *, prior_contacts: int = 0, overdue_business_days: int = 0,
                           damaged: bool = False, item_value: Decimal = Decimal(0),
                           human_requested: bool = False, unauthorized_activity: bool = False,
                           legal_threat: bool = False, safety_risk: bool = False) -> Calculation:
        flags = {"repeat_contact": prior_contacts >= self.rules.prior_contact_threshold,
                 "overdue_refund": overdue_business_days > self.rules.overdue_escalation_days,
                 "high_value_damage": damaged and item_value > self.rules.high_value_threshold,
                 "human_requested": human_requested, "unauthorized_activity": unauthorized_activity,
                 "legal_threat": legal_threat, "safety_risk": safety_risk}
        return Calculation({"reasons": [name for name, active in flags.items() if active],
                            "needs_ticket_before_handover": any(flags.values())}, ("ESC-1", "ESC-2", "DWM-3"))
