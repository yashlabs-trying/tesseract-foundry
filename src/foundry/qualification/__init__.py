from ..models import (
    DeploymentCandidate, PerformanceEvidence, EconomicEvaluation,
    WorkloadProfile, ServiceRecommendation, RejectedReason, RankBy,
    HardwareProfile, EvidenceType,
)
from ..hardware import lookup_gpu, check_precision_support
from typing import Optional
import uuid
from datetime import datetime


class QualificationResult:
    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


class QualificationEngine:
    def __init__(self):
        pass

    def check_hardware_feasibility(self, candidate: DeploymentCandidate, hardware: HardwareProfile):
        hw_spec = lookup_gpu(hardware.gpu_model)
        if not hw_spec:
            return QualificationResult.FAIL, "Unknown GPU model"
        if candidate.gpu_count > hardware.device_count:
            return QualificationResult.FAIL, f"Requires {candidate.gpu_count} GPUs but only {hardware.device_count} available"
        if candidate.tensor_parallelism > candidate.gpu_count:
            return QualificationResult.FAIL, f"TP={candidate.tensor_parallelism} exceeds GPU count={candidate.gpu_count}"
        if candidate.pipeline_parallelism > 1 and candidate.gpu_count < candidate.tensor_parallelism * candidate.pipeline_parallelism:
            return QualificationResult.FAIL, f"TP*PP={candidate.tensor_parallelism * candidate.pipeline_parallelism} exceeds GPU count={candidate.gpu_count}"
        ok, msg = check_precision_support(hardware, candidate.precision)
        if not ok:
            return QualificationResult.FAIL, msg
        return QualificationResult.PASS, None

    def check_vram_constraint(self, candidate: DeploymentCandidate, hardware: HardwareProfile):
        if candidate.expected_vram_gb <= 0:
            return QualificationResult.UNVERIFIED, "No VRAM estimate available"
        if candidate.expected_vram_gb > hardware.vram_per_gpu_gb * 0.9:
            return QualificationResult.FAIL, f"Estimated VRAM {candidate.expected_vram_gb:.1f}GB exceeds 90% of {hardware.vram_per_gpu_gb:.0f}GB available"
        return QualificationResult.PASS, None

    def check_ttft_constraint(self, workload: WorkloadProfile, evidence=None):
        if not workload.warm_latency_target_ms:
            return QualificationResult.PASS, None
        if evidence is None:
            return QualificationResult.UNVERIFIED, "No performance evidence available"
        if evidence.ttft_ms is None:
            return QualificationResult.UNVERIFIED, "No TTFT measurement available"
        if evidence.ttft_ms.p95 is None:
            return QualificationResult.UNVERIFIED, "TTFT P95 missing"
        if evidence.ttft_ms.p95 > workload.warm_latency_target_ms:
            return QualificationResult.FAIL, f"TTFT P95 {evidence.ttft_ms.p95}ms exceeds target {workload.warm_latency_target_ms}ms"
        return QualificationResult.PASS, None

    def check_latency_constraint(self, workload: WorkloadProfile, evidence=None):
        if not workload.warm_latency_target_ms:
            return QualificationResult.PASS, None
        if evidence is None:
            return QualificationResult.UNVERIFIED, "No performance evidence available"
        if evidence.request_latency_ms is None:
            return QualificationResult.UNVERIFIED, "No request latency measurement available"
        if evidence.request_latency_ms.p95 is None:
            return QualificationResult.UNVERIFIED, "Request latency P95 missing"
        if evidence.request_latency_ms.p95 > workload.warm_latency_target_ms:
            return QualificationResult.FAIL, f"Request P95 latency {evidence.request_latency_ms.p95}ms exceeds target {workload.warm_latency_target_ms}ms"
        return QualificationResult.PASS, None

    def check_throughput_constraint(self, workload: WorkloadProfile, evidence=None):
        if evidence is None:
            return QualificationResult.UNVERIFIED, "No performance evidence available"
        if evidence.throughput is None:
            return QualificationResult.UNVERIFIED, "No throughput measurement available"
        if evidence.throughput.completed_requests_per_second is None:
            return QualificationResult.UNVERIFIED, "Throughput measurement missing"
        if evidence.throughput.completed_requests_per_second < workload.request_rate_rps:
            return QualificationResult.FAIL, f"Throughput {evidence.throughput.completed_requests_per_second} rps below required {workload.request_rate_rps} rps"
        return QualificationResult.PASS, None

    def check_evidence_validity(self, evidence=None):
        if evidence is None:
            return QualificationResult.UNVERIFIED, "No evidence collected"
        if not evidence.valid:
            return QualificationResult.FAIL, f"Evidence marked invalid: {evidence.confidence_limitations or 'unknown reason'}"
        if evidence.evidence_type == EvidenceType.synthetic:
            return QualificationResult.UNVERIFIED, "Evidence is synthetic - not validated against real hardware"
        if evidence.evidence_type == EvidenceType.estimated:
            return QualificationResult.UNVERIFIED, "Evidence is estimated - low confidence"
        if evidence.evidence_type == EvidenceType.measured and evidence.sample_count < 10:
            return QualificationResult.UNVERIFIED, f"Only {evidence.sample_count} samples - low statistical confidence"
        return QualificationResult.PASS, None

    def check_margin_constraint(self, econ, min_margin):
        if econ is None:
            return QualificationResult.UNVERIFIED, "No economic evaluation available"
        if econ.margin_pct < min_margin:
            return QualificationResult.FAIL, f"Margin {econ.margin_pct:.1f}% below minimum {min_margin}%"
        if econ.status == "estimated":
            return QualificationResult.UNVERIFIED, "Economics are estimated, not validated"
        return QualificationResult.PASS, None

    def qualify_candidate(self, candidate, hardware, workload, econ=None, evidence=None, min_margin=20.0):
        failures = []
        unverified = []
        checks = [
            ("hardware feasibility", self.check_hardware_feasibility(candidate, hardware)),
            ("VRAM", self.check_vram_constraint(candidate, hardware)),
            ("evidence validity", self.check_evidence_validity(evidence)),
            ("TTFT", self.check_ttft_constraint(workload, evidence)),
            ("latency", self.check_latency_constraint(workload, evidence)),
            ("throughput", self.check_throughput_constraint(workload, evidence)),
            ("margin", self.check_margin_constraint(econ, min_margin)),
        ]
        for name, (result, reason) in checks:
            if result == QualificationResult.FAIL:
                failures.append(f"{name}: {reason}")
            elif result == QualificationResult.UNVERIFIED and reason:
                unverified.append(f"{name}: {reason}")
        if failures:
            return "FAIL", failures + unverified
        if unverified:
            return "UNVERIFIED", unverified
        return "PASS", []

    def rank_candidates(self, candidates, economics, rank_by):
        scored = []
        for c in candidates:
            econ = economics.get(c.candidate_id)
            if rank_by.value == "lowest_cost":
                score = float(econ.total_cost_monthly) if econ else float("inf")
            elif rank_by.value == "highest_margin":
                score = -float(econ.margin_pct) if econ else float("inf")
            elif rank_by.value == "lowest_gpu_count":
                score = float(c.gpu_count)
            elif rank_by.value == "best_throughput":
                score = -float(c.max_batch_size)
            else:
                score = float(econ.total_cost_monthly) if econ else float("inf")
            scored.append((c, score))
        scored.sort(key=lambda x: x[1])
        return scored

    def recommend(self, candidates, hardware_map, economics, workload, evidence_map, rank_by=None, min_margin=20.0):
        if rank_by is None:
            from ..models import RankBy
            rank_by = RankBy.lowest_cost
        feasible = []
        unverified = []
        rejected = []
        for c in candidates:
            hw = hardware_map.get(c.hardware_id)
            if hw is None:
                rejected.append(RejectedReason(candidate_id=c.candidate_id, reason="Hardware profile not found"))
                continue
            econ = economics.get(c.candidate_id)
            ev = evidence_map.get(c.candidate_id)
            result, reasons = self.qualify_candidate(c, hw, workload, econ, ev, min_margin)
            if result == "PASS":
                feasible.append(c)
            elif result == "UNVERIFIED":
                unverified.append(c)
                rejected.append(RejectedReason(candidate_id=c.candidate_id, reason="UNVERIFIED: " + "; ".join(reasons)))
            else:
                rejected.append(RejectedReason(candidate_id=c.candidate_id, reason="; ".join(reasons)))
        if not feasible:
            if unverified:
                ev = evidence_map.get(unverified[0].candidate_id) if unverified else None
                ec = economics.get(unverified[0].candidate_id) if unverified else None
                return ServiceRecommendation(
                    recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                    recommended_candidate_id="NO_VERIFIED_DEPLOYMENT",
                    feasible_candidates=[c.candidate_id for c in unverified],
                    rejected_candidates=rejected,
                    expected_performance=ev,
                    economic_assumptions=ec,
                    approval_status="unverified_only",
                    created_at=datetime.utcnow(),
                )
            return ServiceRecommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                recommended_candidate_id="NO_FEASIBLE_DEPLOYMENT",
                feasible_candidates=[],
                rejected_candidates=rejected,
                approval_status="no_feasible_deployment",
                created_at=datetime.utcnow(),
            )
        ranked = self.rank_candidates(feasible, economics, rank_by)
        best = ranked[0][0]
        return ServiceRecommendation(
            recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
            recommended_candidate_id=best.candidate_id,
            feasible_candidates=[c.candidate_id for c, _ in ranked],
            rejected_candidates=rejected,
            expected_performance=evidence_map.get(best.candidate_id),
            economic_assumptions=economics.get(best.candidate_id),
            deployment_artifact_ref=f"manifest-{best.candidate_id}",
            approval_status="pending_review",
            created_at=datetime.utcnow(),
        )
