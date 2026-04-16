"""
Shared types used by dynamic routing modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional


class TaskType(str, Enum):
    GENERAL = "GENERAL"
    FACT_QA = "FACT_QA"
    INFORMATION_EXTRACTION = "INFORMATION_EXTRACTION"
    SUMMARIZATION = "SUMMARIZATION"
    REASONING = "REASONING"
    MATH = "MATH"
    CODE = "CODE"
    MULTI_HOP_QA = "MULTI_HOP_QA"
    LONG_CONTEXT = "LONG_CONTEXT"
    KNOWLEDGE_INTENSIVE = "KNOWLEDGE_INTENSIVE"
    TOOL_USE = "TOOL_USE"
    DATA_ANALYSIS = "DATA_ANALYSIS"


class ComplexityLevel(IntEnum):
    SIMPLE = 0
    MEDIUM = 1
    COMPLEX = 2


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CostLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def rank(self) -> int:
        return {
            CostLevel.LOW: 0,
            CostLevel.MEDIUM: 1,
            CostLevel.HIGH: 2,
        }[self]


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass
class TaskProfile:
    query: str
    task_type: TaskType
    complexity: ComplexityLevel
    confidence: float
    risk_level: RiskLevel = RiskLevel.LOW
    requires_tools: bool = False
    requires_retrieval: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    paradigm_id: str
    reason: str
    expected_cost: CostLevel
    model_plan: str
    fallback_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    output: str
    confidence: float
    latency_ms: int
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    task_profile: TaskProfile
    initial_decision: RoutingDecision
    final_decision: RoutingDecision
    initial_result: ExecutionResult
    final_result: ExecutionResult
    escalated: bool
    attempts: int
    trace: List[str] = field(default_factory=list)
