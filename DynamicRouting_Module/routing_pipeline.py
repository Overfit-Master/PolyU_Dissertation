"""
动态路由端到端编排（路由 -> 执行 -> 质量门 -> 重试/升级）。
"""

from __future__ import annotations

from typing import List, Optional

from .agent_paradigm import ParadigmPool
from .executor import MockParadigmExecutor, ParadigmExecutor
from .quality_gate import QualityGate, QualityPolicy
from .router_policy import RuleBasedRouter
from .routing_types import (
    ComplexityLevel,
    ExecutionStatus,
    PipelineResult,
    RiskLevel,
    RoutingDecision,
    TaskProfile,
    TaskType,
)


class DynamicRoutingEngine:
    """
    动态路由流水线的编排器：
    1) 路由：根据任务画像选择初始范式。
    2) 执行：按选择的范式执行（默认使用模拟执行器，便于本地验证架构）。
    3) 质量门：基于执行状态/置信度判断是否需要重试或升级。

    备注：
    - 该实现刻意保持“轻量 + 可解释 + 在给定路由器/执行器时可复现”，便于做消融实验与论文复现。
    - `attempts` 统计总执行次数（包含同范式重试与升级后的执行）。
    """

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
        """
        让一个任务画像走完整条动态路由流水线，并返回带 trace 的结果。

        行为说明：
        - 一定会先按初始路由执行至少一次。
        - 若质量门判断需要升级/纠错：
          - 对于执行失败且 `retry_on_failure=True` 的情况，优先在同一范式上重试一次（把失败当作可能的瞬态问题）。
          - 否则沿着路由器的回退链路逐级升级，直到达到 `max_attempts` 或无可用回退范式。
        """
        trace: List[str] = []

        max_attempts = int(self.quality_gate.policy.max_attempts)
        if max_attempts < 1:
            raise ValueError(f"quality_gate.policy.max_attempts must be >= 1, got {max_attempts}")

        initial_decision = self.router.route(profile)
        trace.append(f"initial_route={initial_decision.paradigm_id}")

        attempts = 0
        escalated = False
        decision: RoutingDecision = initial_decision

        # 先占位：首次执行后会被填充（保证返回结果结构稳定）。
        initial_result = None
        final_decision: RoutingDecision = initial_decision
        final_result = None

        while attempts < max_attempts:
            attempts += 1

            result = self.executor.execute(profile, decision)
            if attempts == 1:
                initial_result = result
            final_result = result
            final_decision = decision

            trace.append(
                " | ".join(
                    [
                        f"attempt={attempts}",
                        f"paradigm={decision.paradigm_id}",
                        f"status={result.status.value}",
                        f"conf={result.confidence:.3f}",
                        f"latency_ms={result.latency_ms}",
                    ]
                )
            )

            needs_action, reason = self.quality_gate.needs_escalation(profile, result)
            trace.append(f"quality_check_{attempts}={reason}")
            if not needs_action:
                break

            if attempts >= max_attempts:
                trace.append("max_attempts_reached")
                break

            # 失败可能是瞬态（网络、限流、工具抖动等）：优先同范式重试。
            if result.status == ExecutionStatus.FAILED and self.quality_gate.policy.retry_on_failure:
                trace.append("retry_same_paradigm")
                continue

            next_decision = self.router.escalate(profile, decision)
            if next_decision is None:
                trace.append("no_fallback_available")
                break

            escalated = True
            trace.append(f"escalate_to={next_decision.paradigm_id}")
            decision = next_decision

        # 到这里一定至少执行过一次。
        assert initial_result is not None
        assert final_result is not None

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
