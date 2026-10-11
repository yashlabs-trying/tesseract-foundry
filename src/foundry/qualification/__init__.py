from ..models import (
    DeploymentCandidate, PerformanceEvidence, EconomicEvaluation,
    WorkloadProfile, ServiceRecommendation, RejectedReason, RankBy,
)
from typing import Optional
import uuid
from datetime import datetime

class QualificationEngine:
    def __init__(self):
        pass

    def check_vram_constraint(self, candidate: DeploymentCandidate, workload: WorkloadProfile) -> tuple[bool, Optional[str]]:
        if candidate.expected_vram_gb <= 0:
            return False, "No VRAM estimate available"
        return True, None

    def check_latency_constraint(self, candidate: DeploymentCandidate, workload: WorkloadProfile, evidence: Optional[PerformanceEvidence] = None) -> tuple[bool, Optional[str]]:
        if not workload.warm_latency_target_ms:
            return True, None
        if evidence and evidence.request_latency_ms and evidence.request_latency_ms.p95:
            if evidence.request_latency_ms.p95 > workload.warm_latency_target_ms:
                return False, f"P95 latency {evidence.request_latency_ms.p95}ms exceeds target {workload.warm_latency_target_ms}ms"
        return True, None

    def check_throughput_constraint(self, candidate: DeploymentCandidate, workload: WorkloadProfile, evidence: Optional[PerformanceEvidence] = None) -> tuple[bool, Optional[str]]:
        if evidence and evidence.throughput and evidence.throughput.completed_requests_per_second:
            if evidence.throughput.completed_requests_per_second < workload.request_rate_rps:
                return False, f"Throughput {evidence.throughput.completed_requests_per_second} rps below required {workload.request_rate_rps} rps"
        return True, None

    def check_margin_constraint(self, econ: EconomicEvaluation, min_margin: float) -> tuple[bool, Optional[str]]:
        if econ.margin_pct < min_margin:
            return False, f"Margin {econ.margin_pct:.1f}% below minimum {min_margin}%"
        return True, None

    def qualify_candidate(self, candidate: DeploymentCandidate, workload: WorkloadProfile, econ: Optional[EconomicEvaluation] = None, evidence: Optional[PerformanceEvidence] = None, min_margin: float = 20.0) -> tuple[bool, list[str]]:
        reasons = []
        ok, reason = self.check_vram_constraint(candidate, workload)
        if not ok:
            reasons.append(reason or "VRAM check failed")
        ok, reason = self.check_latency_constraint(candidate, workload, evidence)
        if not ok:
            reasons.append(reason or "Latency check failed")
        ok, reason = self.check_throughput_constraint(candidate, workload, evidence)
        if not ok:
            reasons.append(reason or "Throughput check failed")
        if econ:
            ok, reason = self.check_margin_constraint(econ, min_margin)
            if not ok:
                reasons.append(reason or "Margin check failed")
        return len(reasons) == 0, reasons

    def rank_candidates(self, candidates: list[DeploymentCandidate], economics: dict[str, EconomicEvaluation], rank_by: RankBy) -> list[tuple[DeploymentCandidate, int]]:
        scored = []
        for c in candidates:
            econ = economics.get(c.candidate_id)
            if rank_by == RankBy.lowest_cost:
                score = float(econ.total_cost_monthly) if econ else float("inf")
            elif rank_by == RankBy.highest_margin:
                score = -float(econ.margin_pct) if econ else float("inf")
            elif rank_by == RankBy.lowest_gpu_count:
                score = c.gpu_count
            elif rank_by == RankBy.best_throughput:
                score = -c.max_batch_size
            else:
                score = float(econ.total_cost_monthly) if econ else float("inf")
            scored.append((c, score))
        scored.sort(key=lambda x: x[1])
        return scored

    def recommend(self, candidates: list[DeploymentCandidate], economics: dict[str, EconomicEvaluation], workload: WorkloadProfile, evidence_map: dict[str, PerformanceEvidence], rank_by: RankBy = RankBy.lowest_cost, min_margin: float = 20.0) -> ServiceRecommendation:
        feasible = []
        rejected = []
        for c in candidates:
            econ = economics.get(c.candidate_id)
            ev = evidence_map.get(c.candidate_id)
            ok, reasons = self.qualify_candidate(c, workload, econ, ev, min_margin)
            if ok:
                feasible.append(c)
            else:
                rejected.append(RejectedReason(candidate_id=c.candidate_id, reason="; ".join(reasons)))

        if not feasible:
            return ServiceRecommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                recommended_candidate_id="NO_FEASIBLE_DEPLOYMENT",
                feasible_candidates=[],
                rejected_candidates=rejected,
                approval_status="no_feasible_deployment",
            )

        ranked = self.rank_candidates(feasible, economics, rank_by)
        best = ranked[0][0]
        return ServiceRecommendation(
            recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
            recommended_candidate_id=best.candidate_id,
            feasible_candidates=[c.candidate_id for c,_ in ranked],
            rejected_candidates=rejected,
            expected_performance=evidence_map.get(best.candidate_id),
            economic_assumptions=economics.get(best.candidate_id),
            deployment_artifact_ref=f"manifest-{best.candidate_id}",
            approval_status="pending_review",
            created_at=datetime.utcnow(),
        )
