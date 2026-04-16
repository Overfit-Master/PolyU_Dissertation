"""
End-to-end dynamic routing orchestration.
"""

from __future__ import annotations

from typing import List, Optional

from .agent_paradigm import ParadigmPool
from .executor import MockParadigmExecutor, ParadigmExecutor
from .quality_gate import QualityGate, QualityPolicy
from .router_policy import RuleBasedRouter
from .routing_types import (
    ComplexityLevel,
    PipelineResult,
    RiskLevel,
    RoutingDecision,
    TaskProfile,
    TaskType,
)


class DynamicRoutingEngine:
    """Glue layer: route -> execute -> quality check -> optional escalation."""

    def __init__(
        self,
        paradigm_pool: Optional[ParadigmPool] = None,
        router: Optional[RuleBasedRouter] = None,
        executor: Optional[ParadigmExecutor] = None,
        quality_gate: Optional[QualityGate] = None,
    ) -> None:
        self.paradigm_pool = paradigm_pool or ParadigmPool()
        self.router = router or RuleBasedRouter(paradigm_pool=self.paradigm_pool)
        self.executor = executor or MockParadigmExecutor()
        self.quality_gate = quality_gate or QualityGate()

    def run(self, profile: TaskProfile) -> PipelineResult:
        trace: List[str] = []

        initial_decision = self.router.route(profile)
        trace.append(f"initial_route={initial_decision.paradigm_id}")
        initial_result = self.executor.execute(profile, initial_decision)
        trace.append(f"initial_exec_status={initial_result.status.value}")

        final_decision: RoutingDecision = initial_decision
        final_result = initial_result
        attempts = 1
        escalated = False

        escalate, reason = self.quality_gate.needs_escalation(profile, initial_result)
        trace.append(f"quality_check={reason}")

        if escalate and attempts < self.quality_gate.policy.max_attempts:
            next_decision = self.router.escalate(profile, initial_decision)
            if next_decision is not None:
                escalated = True
                attempts += 1
                final_decision = next_decision
                trace.append(f"escalate_to={next_decision.paradigm_id}")
                final_result = self.executor.execute(profile, next_decision)
                trace.append(f"escalated_exec_status={final_result.status.value}")
                second_check, second_reason = self.quality_gate.needs_escalation(profile, final_result)
                trace.append(f"post_escalation_quality_check={second_reason}")
                if second_check:
                    trace.append("quality_still_low_after_escalation")

        return PipelineResult(
            task_profile=profile,
            initial_decision=initial_decision,
            final_decision=final_decision,
            initial_result=initial_result,
            final_result=final_result,
            escalated=escalated,
            attempts=attempts,
            trace=trace,
        )


def build_task_profile(
    query: str,
    task_type: TaskType = TaskType.REASONING,
    complexity: ComplexityLevel = ComplexityLevel.SIMPLE,
    confidence: float = 0.7,
    risk_level: RiskLevel = RiskLevel.LOW,
    requires_tools: bool = False,
    requires_retrieval: bool = False,
) -> TaskProfile:
    return TaskProfile(
        query=query,
        task_type=task_type,
        complexity=complexity,
        confidence=confidence,
        risk_level=risk_level,
        requires_tools=requires_tools,
        requires_retrieval=requires_retrieval,
    )


def build_task_profile_from_bert(
    query: str,
    bert_complexity_level: int,
    bert_confidence: float,
    risk_level: RiskLevel = RiskLevel.LOW,
    task_type: TaskType = TaskType.REASONING,
) -> TaskProfile:
    """
    Standard mapping for BERT binary output:
    - 0 -> SIMPLE
    - 1 -> COMPLEX
    """
    if bert_complexity_level not in (0, 1):
        raise ValueError(f"bert_complexity_level must be 0 or 1, got {bert_complexity_level}")

    complexity = ComplexityLevel.SIMPLE if bert_complexity_level == 0 else ComplexityLevel.COMPLEX
    return TaskProfile(
        query=query,
        task_type=task_type,
        complexity=complexity,
        confidence=max(0.0, min(1.0, float(bert_confidence))),
        risk_level=risk_level,
        requires_tools=False,
        requires_retrieval=False,
        metadata={
            "bert_complexity_level": bert_complexity_level,
            "bert_confidence": bert_confidence,
        },
    )


if __name__ == "__main__":
    engine = DynamicRoutingEngine(quality_gate=QualityGate(QualityPolicy(min_confidence=0.80)))
    profile = build_task_profile_from_bert(
        query="If 5 machines make 5 widgets in 5 minutes, how long would 100 machines make 100 widgets?",
        bert_complexity_level=1,
        bert_confidence=0.62,
        task_type=TaskType.MATH,
        risk_level=RiskLevel.MEDIUM,
    )
    result = engine.run(profile)
    print("Final paradigm:", result.final_decision.paradigm_id)
    print("Final confidence:", f"{result.final_result.confidence:.3f}")
    print("Escalated:", result.escalated)
    print("Trace:", result.trace)
