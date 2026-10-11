from ..models import DeploymentCandidate, PerformanceEvidence, EvidenceType, MetricAggregate, ThroughputMetrics
from typing import Optional
import uuid

class FixtureProfilerAdapter:
    def submit_job(self, candidate: DeploymentCandidate) -> str:
        return f"fixture-job-{uuid.uuid4().hex[:8]}"

    def get_results(self, candidate: DeploymentCandidate) -> PerformanceEvidence:
        params = candidate.model_id.count("0") * 100_000_000 if "B" in candidate.model_id else 600_000_000
        if candidate.model_id.count("B") > 0 or "B" in candidate.model_id:
            params = 8_000_000_000
        if "141" in candidate.model_id or "22B" in candidate.model_id:
            params = 141_000_000_000
        if "0.6B" in candidate.model_id or "600" in candidate.model_id:
            params = 600_000_000

        if params < 1_000_000_000:
            throughput = 50.0
            ttft = 50.0
            itl = 5.0
            latency = 200.0
        elif params < 10_000_000_000:
            throughput = 10.0
            ttft = 150.0
            itl = 15.0
            latency = 500.0
        else:
            throughput = 2.0
            ttft = 500.0
            itl = 40.0
            latency = 2000.0

        gpu_factor = max(1, candidate.gpu_count)
        throughput *= gpu_factor
        ttft /= gpu_factor * 0.5

        return PerformanceEvidence(
            evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
            candidate_id=candidate.candidate_id,
            evidence_type=EvidenceType.synthetic,
            benchmark_tool="fixture",
            model_revision=candidate.model_id,
            ttft_ms=MetricAggregate(p50=ttft * 0.5, p95=ttft, p99=ttft * 1.5, mean=ttft, unit="ms"),
            itl_ms=MetricAggregate(p50=itl * 0.7, p95=itl, p99=itl * 1.5, mean=itl, unit="ms"),
            request_latency_ms=MetricAggregate(p50=latency * 0.5, p95=latency, p99=latency * 2, mean=latency, unit="ms"),
            throughput=ThroughputMetrics(completed_requests_per_second=throughput),
            sample_count=100,
            valid=True,
        )
