"""
Rule-based router for paradigm selection and escalation.
"""

from __future__ import annotations

from typing import Dict, Optional

from .agent_paradigm import ParadigmPool
from .routing_types import (
    ComplexityLevel,
    CostLevel,
    RiskLevel,
    RoutingDecision,
    TaskProfile,
    TaskType,
)


class RuleBasedRouter:
    """
    A simple but explicit router:
    - route(profile) picks initial paradigm
    - escalate(profile, current_decision) picks next fallback paradigm
    """

    def __init__(
        self,
        paradigm_pool: ParadigmPool,
        low_confidence_threshold: float = 0.65,
        complex_confidence_threshold: float = 0.75,
    ) -> None:
        self.paradigm_pool = paradigm_pool
        self.low_confidence_threshold = low_confidence_threshold
        self.complex_confidence_threshold = complex_confidence_threshold
        self._fallback_graph: Dict[str, str] = {
            "P01_Reasoning_FastPath": "P02_Reasoning_Solver_Verifier",
            "P02_Reasoning_Solver_Verifier": "P03_Reasoning_Decompose_Aggregate",
            "P03_Reasoning_Decompose_Aggregate": "P04_Reasoning_Safe_Escalation",
        }

    def route(self, profile: TaskProfile) -> RoutingDecision:
        paradigm_id = self._select_paradigm_id(profile)
        return self._build_decision(profile=profile, paradigm_id=paradigm_id)

    def escalate(self, profile: TaskProfile, current: RoutingDecision) -> Optional[RoutingDecision]:
        next_id = self._fallback_graph.get(current.paradigm_id)
        if next_id is None:
            return None
        return self._build_decision(profile=profile, paradigm_id=next_id, escalated_from=current.paradigm_id)

    def _select_paradigm_id(self, profile: TaskProfile) -> str:
        reasoning_types = {TaskType.REASONING, TaskType.MATH, TaskType.MULTI_HOP_QA}
        if profile.risk_level == RiskLevel.HIGH:
            return "P04_Reasoning_Safe_Escalation"

        # This project version is intentionally constrained to reasoning-only experiments.
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
            expected_cost=CostLevel(paradigm.cost_level.value),
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
        chain: list[str] = []
        current = paradigm_id
        visited: set[str] = set()
        while current in self._fallback_graph and current not in visited:
            visited.add(current)
            current = self._fallback_graph[current]
            chain.append(current)
        return chain
