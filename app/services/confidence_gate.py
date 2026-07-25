from dataclasses import dataclass
from typing import Literal

from app.core.config import RCA_CONFIDENCE_THRESHOLD


GateDecision = Literal["deliver", "abstain", "rerun"]


@dataclass
class GateResult:
    decision: GateDecision
    reason: str


class ConfidenceGate:
    def __init__(self, threshold: float = RCA_CONFIDENCE_THRESHOLD):
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        self.threshold = threshold

    def evaluate(self, rca: dict) -> GateResult:
        # 模型主动表示无法判断
        if rca.get("abstained") is True:
            return GateResult(
                decision="abstain",
                reason="The reasoner abstained.",
            )

        failure_class = rca.get("failure_class", "unknown")

        # 非法类别不允许发布
        if failure_class == "unknown":
            return GateResult(
                decision="abstain",
                reason="Failure class is unknown.",
            )

        confidence = rca.get("confidence")

        if not isinstance(confidence, (int, float)):
            return GateResult(
                decision="abstain",
                reason="Confidence is missing or invalid.",
            )

        if not 0 <= confidence <= 1:
            return GateResult(
                decision="abstain",
                reason="Confidence is outside the range 0..1.",
            )

        if confidence < self.threshold:
            return GateResult(
                decision="abstain",
                reason=(
                    f"Confidence {confidence:.2f} is below "
                    f"threshold {self.threshold:.2f}."
                ),
            )

        # flaky 不能直接发布结论，必须先 rerun
        if failure_class == "flaky":
            return GateResult(
                decision="rerun",
                reason="Suspected flaky failure requires a rerun.",
            )

        return GateResult(
            decision="deliver",
            reason="RCA is grounded and confidence is sufficient.",
        )