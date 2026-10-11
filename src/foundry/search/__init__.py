from ..models import (
    DeploymentCandidate, HardwareProfile, ModelSpec, WorkloadProfile,
    PerformanceEvidence, EconomicEvaluation, RankBy, BackendType, PrecisionType,
)
from ..candidates import FixtureCandidateProvider
from ..economics import evaluate
from ..profiling import FixtureProfilerAdapter
from typing import Optional, Protocol
import itertools


class ConfigurationSearch:
    def __init__(self):
        self._provider = FixtureCandidateProvider()
        self._profiler = FixtureProfilerAdapter()

    def generate_all_candidates(
        self,
        model: ModelSpec,
        hardware_list: list[HardwareProfile],
        workload: WorkloadProfile,
        backends: Optional[list[BackendType]] = None,
        precisions: Optional[list[PrecisionType]] = None,
    ) -> list[DeploymentCandidate]:
        candidates = self._provider.generate_candidates(model, hardware_list, workload)
        return candidates

    def search_and_rank(
        self,
        model: ModelSpec,
        hardware_list: list[HardwareProfile],
        workload: WorkloadProfile,
        input_price: float = 0.15,
        output_price: float = 0.60,
        rank_by: str = "lowest_cost",
        min_margin: float = 20.0,
    ) -> list[dict]:
        from decimal import Decimal
        from ..qualification import QualificationEngine, QualificationResult

        candidates = self.generate_all_candidates(model, hardware_list, workload)
        hardware_map = {h.hardware_id: h for h in hardware_list}
        inp = Decimal(str(input_price))
        outp = Decimal(str(output_price))

        results = []
        for c in candidates:
            hw = hardware_map.get(c.hardware_id)
            if hw is None:
                continue
            econ = evaluate(c, workload, hw, inp, outp)
            ev = self._profiler.get_results(c)

            engine = QualificationEngine()
            result, reasons = engine.qualify_candidate(c, hw, workload, econ, ev, min_margin)

            results.append({
                "candidate": c,
                "hardware": hw,
                "economics": econ,
                "evidence": ev,
                "qualification": result,
                "reasons": reasons,
            })

        if rank_by == "margin":
            results.sort(key=lambda r: r["economics"].margin_pct if r["economics"] else 0, reverse=True)
        elif rank_by == "throughput":
            results.sort(key=lambda r: r["evidence"].throughput.completed_requests_per_second if r["evidence"] and r["evidence"].throughput else 0, reverse=True)
        elif rank_by == "latency":
            results.sort(key=lambda r: r["evidence"].request_latency_ms.p95 if r["evidence"] and r["evidence"].request_latency_ms else float("inf"))
        else:
            results.sort(key=lambda r: float(r["economics"].total_cost_monthly) if r["economics"] else float("inf"))

        return results

    def select_sku(
        self,
        model: ModelSpec,
        hardware_list: list[HardwareProfile],
        workload: WorkloadProfile,
        input_price: float = 0.15,
        output_price: float = 0.60,
        min_margin: float = 20.0,
    ) -> dict:
        results = self.search_and_rank(model, hardware_list, workload, input_price, output_price, "lowest_cost", min_margin)
        feasible = [r for r in results if r["qualification"] == "PASS"]
        if not feasible:
            return {
                "selected": None,
                "all_results": results,
                "message": "No feasible configuration found",
            }
        best = feasible[0]
        return {
            "selected": best,
            "all_results": results,
            "message": f"Selected cheapest feasible: {best['candidate'].candidate_id} on {best['hardware'].gpu_model}",
            "reasoning": [
                f"Hardware: {best['hardware'].gpu_model} ({best['hardware'].vram_per_gpu_gb}GB, {best['hardware'].cost_per_gpu_hour_usd}/hr)",
                f"Engine: {best['candidate'].serving_engine.value} TP={best['candidate'].tensor_parallelism}",
                f"Monthly Cost: ${float(best['economics'].total_cost_monthly):,.2f}",
                f"Monthly Revenue: ${float(best['economics'].estimated_revenue_monthly):,.2f}",
                f"Margin: {best['economics'].margin_pct:.1f}%",
                f"Throughput: {best['evidence'].throughput.completed_requests_per_second if best['evidence'] and best['evidence'].throughput else 'N/A'} rps",
            ],
        }
