"""
路由后执行层的抽象定义（用于把“路由决策”落实为具体执行）。
"""

from __future__ import annotations

import random
from time import perf_counter
from typing import Protocol

from .routing_types import ExecutionResult, ExecutionStatus, RoutingDecision, TaskProfile


class ParadigmExecutor(Protocol):
    def execute(self, profile: TaskProfile, decision: RoutingDecision) -> ExecutionResult:
        """执行一次路由决策，并返回执行结果。"""


class MockParadigmExecutor:
    """
    本地模拟执行器：用于验证路由/质量门等架构逻辑。
    若要接入真实大模型或工具调用，请在实验代码中替换为真实的执行器实现。
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._base_confidence = {
            "P01_Reasoning_FastPath": 0.78,
            "P02_Reasoning_Solver_Verifier": 0.86,
            "P03_Reasoning_Decompose_Aggregate": 0.88,
            "P04_Reasoning_Safe_Escalation": 0.90,
        }
        self._base_latency_ms = {
            "P01_Reasoning_FastPath": 420,
            "P02_Reasoning_Solver_Verifier": 1050,
            "P03_Reasoning_Decompose_Aggregate": 1500,
            "P04_Reasoning_Safe_Escalation": 1850,
        }

    def execute(self, profile: TaskProfile, decision: RoutingDecision) -> ExecutionResult:
        start = perf_counter()

        base = self._base_confidence.get(decision.paradigm_id, 0.70)
        complexity_penalty = float(profile.complexity) * 0.06
        jitter = self._rng.uniform(-0.03, 0.03)
        confidence = max(0.05, min(0.99, base - complexity_penalty + jitter))

        failure_prob = 0.01 + (0.05 if confidence < 0.55 else 0.0)
        failed = self._rng.random() < failure_prob
        status = ExecutionStatus.FAILED if failed else ExecutionStatus.SUCCESS

        latency = self._base_latency_ms.get(decision.paradigm_id, 1000) + int(self._rng.uniform(-80, 120))
        prompt_tokens = max(20, int(len(profile.query) / 2.5))
        completion_tokens = 120 if status == ExecutionStatus.SUCCESS else 0
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        elapsed_ms = int((perf_counter() - start) * 1000)
        output = (
            f"[{decision.paradigm_id}] simulated response for query: {profile.query[:80]}"
            if status == ExecutionStatus.SUCCESS
            else ""
        )
        error = None if status == ExecutionStatus.SUCCESS else "Simulated execution failure."
        return ExecutionResult(
            status=status,
            output=output,
            confidence=confidence,
            latency_ms=max(1, latency + elapsed_ms),
            token_usage=token_usage,
            metadata={
                "model_plan": decision.model_plan,
                "paradigm_reason": decision.reason,
            },
            error=error,
        )
