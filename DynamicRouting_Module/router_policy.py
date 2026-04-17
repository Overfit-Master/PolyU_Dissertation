"""
基于规则的路由器：用于选择范式以及升级策略。
"""

from __future__ import annotations

from typing import Dict, Optional

from .agent_paradigm import ParadigmPool
from .routing_types import (
    ComplexityLevel,
    RiskLevel,
    RoutingDecision,
    TaskProfile,
    TaskType,
)


class RuleBasedRouter:
    """
    一个简单但“显式可解释”的路由器：
    - `route(profile)`：选择初始范式
    - `escalate(profile, current_decision)`：沿回退链路选择下一个范式

    设计意图：
    - 路由逻辑保持透明，便于改规则做消融实验。
    - 本论文原型优先使用规则路由（避免训练路由器带来的额外成本与不确定性）。
    - 使用固定回退图，让升级路径确定、可审计、便于复盘。
    """

    def __init__(
        self,
        paradigm_pool: ParadigmPool,
        low_confidence_threshold: float = 0.65,
        complex_confidence_threshold: float = 0.75,
    ) -> None:
        # 阈值的含义是“置信度足够高则停留在更便宜的范式上”。
        # 例如：SIMPLE 且 confidence >= low_confidence_threshold 时使用 FastPath。
        self.paradigm_pool = paradigm_pool
        self.low_confidence_threshold = low_confidence_threshold
        self.complex_confidence_threshold = complex_confidence_threshold
        self._fallback_graph: Dict[str, str] = {
            "P01_Reasoning_FastPath": "P02_Reasoning_Solver_Verifier",
            "P02_Reasoning_Solver_Verifier": "P03_Reasoning_Decompose_Aggregate",
            "P03_Reasoning_Decompose_Aggregate": "P04_Reasoning_Safe_Escalation",
        }

    def route(self, profile: TaskProfile) -> RoutingDecision:
        """根据任务画像选择初始范式。"""
        paradigm_id = self._select_paradigm_id(profile)
        return self._build_decision(profile=profile, paradigm_id=paradigm_id)

    def escalate(self, profile: TaskProfile, current: RoutingDecision) -> Optional[RoutingDecision]:
        """沿回退图升级到下一范式（若存在）。"""
        next_id = self._fallback_graph.get(current.paradigm_id)
        if next_id is None:
            return None
        return self._build_decision(profile=profile, paradigm_id=next_id, escalated_from=current.paradigm_id)

    def _select_paradigm_id(self, profile: TaskProfile) -> str:
        """
        确定性地选择一个范式 id（`paradigm_id`）。

        当前约束：本论文原型先聚焦于推理类任务（推理/数学/多跳问答）。
        非推理任务统一路由到“最安全/最高容量”的范式，避免在未覆盖能力范围内静默劣化。
        """
        reasoning_types = {TaskType.REASONING, TaskType.MATH, TaskType.MULTI_HOP_QA}
        if profile.risk_level == RiskLevel.HIGH:
            return "P04_Reasoning_Safe_Escalation"

        # 论文原型阶段刻意限制为“仅推理类任务”的实验设置。
        if profile.task_type not in reasoning_types:
            return "P04_Reasoning_Safe_Escalation"

        if profile.complexity == ComplexityLevel.SIMPLE:
            if profile.confidence >= self.low_confidence_threshold:
                return "P01_Reasoning_FastPath"
            return "P02_Reasoning_Solver_Verifier"

        if profile.complexity == ComplexityLevel.COMPLEX:
            if profile.confidence >= self.complex_confidence_threshold:
                return "P02_Reasoning_Solver_Verifier"
            return "P03_Reasoning_Decompose_Aggregate"

        # MEDIUM is kept for compatibility but should be rare in BERT(0/1) setting.
        if profile.complexity == ComplexityLevel.MEDIUM:
            return "P02_Reasoning_Solver_Verifier"

        candidates = self.paradigm_pool.find_candidates(
            task_type=profile.task_type,
            complexity=profile.complexity,
            requires_tools=profile.requires_tools,
        )
        if candidates:
            return candidates[0].paradigm_id
        return "P04_Reasoning_Safe_Escalation"

    def _build_decision(
        self,
        profile: TaskProfile,
        paradigm_id: str,
        escalated_from: Optional[str] = None,
    ) -> RoutingDecision:
        paradigm = self.paradigm_pool.get_paradigm_by_id(paradigm_id)
        reason_parts = [
            f"task_type={profile.task_type.value}",
            f"complexity={int(profile.complexity)}",
            f"task_conf={profile.confidence:.2f}",
            f"risk={profile.risk_level.value}",
        ]
        if profile.requires_tools:
            reason_parts.append("requires_tools=True")
        if profile.requires_retrieval:
            reason_parts.append("requires_retrieval=True")
        if escalated_from:
            reason_parts.append(f"escalated_from={escalated_from}")

        fallback = self._collect_fallback_chain(paradigm_id)
        return RoutingDecision(
            paradigm_id=paradigm.paradigm_id,
            reason=" | ".join(reason_parts),
            # `Paradigm.cost_level` 本身就是 `CostLevel`，无需再做枚举转换。
            expected_cost=paradigm.cost_level,
            model_plan=paradigm.recommended_scale,
            fallback_chain=fallback,
            metadata={
                "paradigm_name": paradigm.name,
                "use_verifier": paradigm.use_verifier,
                "use_parallel_workers": paradigm.use_parallel_workers,
                "max_sub_agents": paradigm.max_sub_agents,
            },
        )

    def _collect_fallback_chain(self, paradigm_id: str) -> list[str]:
        """从给定范式开始收集所有下游 fallback（带环路保护）。"""
        chain: list[str] = []
        current = paradigm_id
        visited: set[str] = set()
        while current in self._fallback_graph and current not in visited:
            visited.add(current)
            current = self._fallback_graph[current]
            chain.append(current)
        return chain
