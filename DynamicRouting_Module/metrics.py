"""
动态路由实验的简单指标记录器。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .routing_types import PipelineResult


class RoutingMetricsRecorder:
    def __init__(self) -> None:
        self._records: List[PipelineResult] = []

    def record(self, result: PipelineResult) -> None:
        self._records.append(result)

    def aggregate(self) -> Dict[str, float]:
        if not self._records:
            return {
                "runs": 0,
                "escalation_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
                "avg_total_tokens": 0.0,
            }

        runs = len(self._records)
        escalated = sum(1 for item in self._records if item.escalated)
        avg_confidence = sum(item.final_result.confidence for item in self._records) / runs
        avg_latency = sum(item.final_result.latency_ms for item in self._records) / runs
        avg_tokens = sum(item.final_result.token_usage.get("total_tokens", 0) for item in self._records) / runs
        return {
            "runs": float(runs),
            "escalation_rate": escalated / runs,
            "avg_confidence": avg_confidence,
            "avg_latency_ms": avg_latency,
            "avg_total_tokens": avg_tokens,
        }

    def save_jsonl(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for item in self._records:
                file.write(json.dumps(self._to_json_dict(item), ensure_ascii=False) + "\n")

    def _to_json_dict(self, item: PipelineResult) -> Dict[str, object]:
        data = asdict(item)
        data["task_profile"]["task_type"] = item.task_profile.task_type.value
        data["task_profile"]["complexity"] = int(item.task_profile.complexity)
        data["task_profile"]["risk_level"] = item.task_profile.risk_level.value
        data["initial_decision"]["expected_cost"] = item.initial_decision.expected_cost.value
        data["final_decision"]["expected_cost"] = item.final_decision.expected_cost.value
        data["initial_result"]["status"] = item.initial_result.status.value
        data["final_result"]["status"] = item.final_result.status.value
        return data
