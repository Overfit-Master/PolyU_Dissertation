"""
Paradigm registry for dynamic routing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from .routing_types import ComplexityLevel, CostLevel, TaskType


@dataclass(frozen=True)
class Paradigm:
    """Data structure of one routing paradigm."""

    paradigm_id: str
    name: str
    description: str
    recommended_scale: str
    cost_level: CostLevel
    supported_task_types: List[TaskType]
    min_complexity: ComplexityLevel
    max_complexity: ComplexityLevel
    requires_tools: bool = False
    use_verifier: bool = False
    use_parallel_workers: bool = False
    max_sub_agents: int = 1
    tags: List[str] = field(default_factory=list)

    def supports(self, task_type: TaskType, complexity: ComplexityLevel, requires_tools: bool) -> bool:
        if requires_tools and not self.requires_tools:
            return False
        if not (self.min_complexity <= complexity <= self.max_complexity):
            return False
        return task_type in self.supported_task_types or TaskType.GENERAL in self.supported_task_types

    def to_prompt_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["cost_level"] = self.cost_level.value
        payload["supported_task_types"] = [task_type.value for task_type in self.supported_task_types]
        payload["min_complexity"] = int(self.min_complexity)
        payload["max_complexity"] = int(self.max_complexity)
        return payload


class ParadigmPool:
    """Central registry and query interface for paradigms."""

    def __init__(self) -> None:
        self._paradigms: Dict[str, Paradigm] = {}
        self._initialize_default_pool()

    def _initialize_default_pool(self) -> None:
        defaults = [
            Paradigm(
                paradigm_id="P01_Reasoning_FastPath",
                name="Reasoning_FastPath_Single",
                description="Single model reasoning path for BERT=0 (simple) tasks.",
                recommended_scale="qwen-turbo",
                cost_level=CostLevel.LOW,
                supported_task_types=[
                    TaskType.REASONING,
                    TaskType.MATH,
                ],
                min_complexity=ComplexityLevel.SIMPLE,
                max_complexity=ComplexityLevel.SIMPLE,
            ),
            Paradigm(
                paradigm_id="P02_Reasoning_Solver_Verifier",
                name="Reasoning_Solver_Verifier",
                description="Primary path for BERT=1 (complex) reasoning with verification.",
                recommended_scale="solver:qwen-plus verifier:qwen-turbo",
                cost_level=CostLevel.MEDIUM,
                supported_task_types=[
                    TaskType.REASONING,
                    TaskType.MATH,
                    TaskType.MULTI_HOP_QA,
                ],
                min_complexity=ComplexityLevel.MEDIUM,
                max_complexity=ComplexityLevel.COMPLEX,
                use_verifier=True,
                max_sub_agents=2,
                tags=["reasoning", "verification"],
            ),
            Paradigm(
                paradigm_id="P03_Reasoning_Decompose_Aggregate",
                name="Reasoning_Decompose_Aggregate",
                description="Escalation path for low-confidence complex reasoning with parallel decomposition.",
                recommended_scale="workers:qwen-plus aggregator:qwen-max",
                cost_level=CostLevel.HIGH,
                supported_task_types=[
                    TaskType.REASONING,
                    TaskType.MATH,
                    TaskType.MULTI_HOP_QA,
                ],
                min_complexity=ComplexityLevel.MEDIUM,
                max_complexity=ComplexityLevel.COMPLEX,
                use_verifier=True,
                use_parallel_workers=True,
                max_sub_agents=4,
                tags=["reasoning", "parallel", "escalation"],
            ),
            Paradigm(
                paradigm_id="P04_Reasoning_Safe_Escalation",
                name="Reasoning_Safe_Escalation",
                description="Final fallback for high-risk or repeatedly low-confidence reasoning outputs.",
                recommended_scale="qwen-max + verifier",
                cost_level=CostLevel.HIGH,
                supported_task_types=[
                    TaskType.REASONING,
                    TaskType.MATH,
                    TaskType.MULTI_HOP_QA,
                ],
                min_complexity=ComplexityLevel.SIMPLE,
                max_complexity=ComplexityLevel.COMPLEX,
                use_verifier=True,
                max_sub_agents=2,
                tags=["reasoning", "fallback", "safety"],
            ),
        ]
        for paradigm in defaults:
            self.register(paradigm)

    def register(self, paradigm: Paradigm, allow_override: bool = False) -> None:
        if paradigm.paradigm_id in self._paradigms and not allow_override:
            raise ValueError(f"Duplicate paradigm id: {paradigm.paradigm_id}")
        self._paradigms[paradigm.paradigm_id] = paradigm

    def list_paradigms(self) -> List[Paradigm]:
        return list(self._paradigms.values())

    def list_ids(self) -> List[str]:
        return list(self._paradigms.keys())

    def get_paradigm_by_id(self, paradigm_id: str) -> Paradigm:
        if paradigm_id not in self._paradigms:
            raise ValueError(f"Unknown paradigm id: {paradigm_id}")
        return self._paradigms[paradigm_id]

    def find_candidates(
        self,
        task_type: TaskType,
        complexity: ComplexityLevel,
        requires_tools: bool = False,
    ) -> List[Paradigm]:
        candidates = [
            paradigm
            for paradigm in self._paradigms.values()
            if paradigm.supports(task_type=task_type, complexity=complexity, requires_tools=requires_tools)
        ]
        candidates.sort(
            key=lambda paradigm: (
                paradigm.cost_level.rank,
                0 if paradigm.use_parallel_workers else 1,
                0 if paradigm.use_verifier else 1,
            )
        )
        return candidates

    def get_prompt_context(
        self,
        compact: bool = True,
        selected_ids: Optional[Iterable[str]] = None,
    ) -> str:
        if selected_ids is None:
            paradigms = self.list_paradigms()
        else:
            paradigms = [self.get_paradigm_by_id(paradigm_id) for paradigm_id in selected_ids]

        payload = [paradigm.to_prompt_dict() for paradigm in paradigms]
        if compact:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pool = ParadigmPool()
    print("Registered paradigms:", pool.list_ids())
    print("Prompt context:")
    print(pool.get_prompt_context(compact=False))
