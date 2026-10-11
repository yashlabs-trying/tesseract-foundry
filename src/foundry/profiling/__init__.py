from ..models import DeploymentCandidate, PerformanceEvidence, EvidenceType, MetricAggregate, ThroughputMetrics
from typing import Optional
import uuid


class ProfilerAdapter:
    def submit_job(self, candidate: DeploymentCandidate) -> str:
        raise NotImplementedError

    def get_results(self, candidate: DeploymentCandidate) -> Optional[PerformanceEvidence]:
        raise NotImplementedError


class FixtureProfilerAdapter(ProfilerAdapter):
    def submit_job(self, candidate: DeploymentCandidate) -> str:
        return f"fixture-job-{uuid.uuid4().hex[:8]}"

    def get_results(self, candidate: DeploymentCandidate) -> PerformanceEvidence:
        model_id = candidate.model_id
        if "0.6B" in model_id or "600M" in model_id:
            params = 600_000_000
        elif "8B" in model_id or "7B" in model_id:
            params = 8_000_000_000
        elif "22B" in model_id or "70B" in model_id:
            params = 141_000_000_000
        else:
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
        ttft /= max(1, gpu_factor // 2 + 1)

        return PerformanceEvidence(
            evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
            candidate_id=candidate.candidate_id,
            evidence_type=EvidenceType.synthetic,
            benchmark_tool="fixture",
            model_revision=candidate.model_id,
            ttft_ms=MetricAggregate(p50=ttft * 0.5, p95=ttft, p99=ttft * 2.0, mean=ttft, min=ttft * 0.3, max=ttft * 3.0, unit="ms"),
            itl_ms=MetricAggregate(p50=itl * 0.7, p95=itl, p99=itl * 2.0, mean=itl, min=itl * 0.5, max=itl * 3.0, unit="ms"),
            request_latency_ms=MetricAggregate(p50=latency * 0.5, p95=latency, p99=latency * 2.0, mean=latency, min=latency * 0.3, max=latency * 3.0, unit="ms"),
            throughput=ThroughputMetrics(completed_requests_per_second=throughput),
            sample_count=100,
            valid=True,
        )


class DynamoProfilerAdapter(ProfilerAdapter):
    DYNAMO_API_VERSION = "nvidia.com/v1alpha1"
    DGDR_KIND = "DynamoGraphDeploymentRequest"

    def __init__(self, api_url: Optional[str] = None, namespace: str = "default"):
        self.api_url = api_url
        self.namespace = namespace
        self._available = api_url is not None

    def is_available(self) -> bool:
        return self._available

    def generate_dgdr(self, candidate: DeploymentCandidate) -> dict:
        return {
            "apiVersion": self.DYNAMO_API_VERSION,
            "kind": self.DGDR_KIND,
            "metadata": {
                "name": f"foundry-{candidate.candidate_id[:16]}",
                "namespace": self.namespace,
            },
            "spec": {
                "model": {
                    "id": candidate.model_id,
                    "revision": getattr(candidate, 'model_revision', "latest"),
                },
                "hardware": {
                    "gpuCount": candidate.gpu_count,
                    "tensorParallelism": candidate.tensor_parallelism,
                    "pipelineParallelism": candidate.pipeline_parallelism,
                },
                "engine": {
                    "name": candidate.serving_engine.value,
                    "version": candidate.engine_version,
                    "precision": candidate.precision.value,
                },
                "autoApply": False,
                "profiling": {
                    "strategy": "thorough",
                    "timeoutSeconds": 3600,
                },
            },
        }

    def submit_dgdr(self, candidate: DeploymentCandidate) -> Optional[str]:
        if not self._available:
            return None
        dgdr = self.generate_dgdr(candidate)
        import json
        return json.dumps(dgdr)

    def extract_candidates(self, raw: dict) -> list[DeploymentCandidate]:
        return []

    def parse_results(self, raw: dict) -> Optional[PerformanceEvidence]:
        if not raw:
            return None
        return PerformanceEvidence(
            evidence_id=f"dynamo-ev-{uuid.uuid4().hex[:8]}",
            candidate_id=raw.get("candidate_id", "unknown"),
            evidence_type=EvidenceType.measured if raw.get("status") == "completed" else EvidenceType.estimated,
            benchmark_tool="dynamo-profiler",
            benchmark_version=self.DYNAMO_API_VERSION,
            model_revision=raw.get("model_revision", ""),
            sample_count=raw.get("sample_count", 0),
            valid=raw.get("status") == "completed",
        )

    def submit_job(self, candidate: DeploymentCandidate) -> str:
        return self.submit_dgdr(candidate) or f"dry-run-{uuid.uuid4().hex[:8]}"

    def get_results(self, candidate: DeploymentCandidate) -> Optional[PerformanceEvidence]:
        return None


class AIPerfAdapter(ProfilerAdapter):
    def __init__(self, binary_path: Optional[str] = None):
        self.binary_path = binary_path
        import os as _os
        self._available = binary_path is not None and _os.path.isfile(binary_path) if binary_path else False

    def is_available(self) -> bool:
        return self._available

    def build_command(self, candidate, model="", concurrency=1, input_len=512, output_len=256):
        return [
            self.binary_path, "--model", model,
            "--backend", candidate.serving_engine.value,
            "--concurrency", str(concurrency),
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            "--num-prompts", "100",
            "--profile-output", "json",
        ] if self._available else []

    def parse_aiperf_output(self, raw_output: str) -> PerformanceEvidence:
        import json
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            data = {"result": "parse_error"}
        return PerformanceEvidence(
            evidence_id=f"aiperf-{uuid.uuid4().hex[:8]}",
            candidate_id=data.get("candidate_id", "unknown"),
            evidence_type=EvidenceType.measured if data.get("status") == "success" else EvidenceType.estimated,
            benchmark_tool="aiperf",
            benchmark_version="1.0",
            ttft_ms=MetricAggregate(
                p50=data.get("ttft_p50"), p95=data.get("ttft_p95"),
                mean=data.get("ttft_mean"), unit="ms",
            ) if data.get("ttft_p50") else None,
            itl_ms=MetricAggregate(
                p50=data.get("itl_p50"), p95=data.get("itl_p95"),
                mean=data.get("itl_mean"), unit="ms",
            ) if data.get("itl_p50") else None,
            throughput=ThroughputMetrics(
                completed_requests_per_second=data.get("throughput"),
                input_tokens_per_second=data.get("input_tokens_per_second"),
                output_tokens_per_second=data.get("output_tokens_per_second"),
            ) if data.get("throughput") else None,
            sample_count=data.get("sample_count", 0),
            valid=data.get("status") == "success",
        )

    def submit_job(self, candidate: DeploymentCandidate) -> str:
        return f"aiperf-job-{uuid.uuid4().hex[:8]}"

    def get_results(self, candidate: DeploymentCandidate) -> Optional[PerformanceEvidence]:
        return None
