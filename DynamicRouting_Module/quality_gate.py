"""
Quality assessment and escalation policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .routing_types import ExecutionResult, ExecutionStatus, RiskLevel, TaskProfile


@dataclass
class QualityPolicy:
    min_confidence: float = 0.72
    high_risk_min_confidence: float = 0.82
    retry_on_failure: bool = True
    max_attempts: int = 2


class QualityGate:
    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self.policy = policy or QualityPolicy()

    def needs_escalation(self, profile: TaskProfile, result: ExecutionResult) -> Tuple[bool, str]:
        if result.status == ExecutionStatus.FAILED:
            if self.policy.retry_on_failure:
                return True, "execution_failed"
            return False, "execution_failed_no_retry"

        threshold = (
            self.policy.high_risk_min_confidence
            if profile.risk_level == RiskLevel.HIGH
            else self.policy.min_confidence
        )
        if result.confidence < threshold:
            return True, f"low_confidence<{threshold:.2f}"
        return False, "quality_ok"
