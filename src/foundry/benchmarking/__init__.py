from ..models import DeploymentCandidate, PerformanceEvidence, WorkloadProfile, EvidenceType
from ..profiling import FixtureProfilerAdapter
from typing import Optional, Protocol


class BenchmarkRunner(Protocol):
    def run(self, candidate: DeploymentCandidate, workload: WorkloadProfile) -> PerformanceEvidence: ...


class FixtureRunner:
    def run(self, candidate: DeploymentCandidate, workload: WorkloadProfile) -> PerformanceEvidence:
        adapter = FixtureProfilerAdapter()
        return adapter.get_results(candidate)


def get_runner(mode: str = "fixture") -> BenchmarkRunner:
    if mode == "aiperf":
        raise NotImplementedError("AIPerfRunner not yet implemented")
    return FixtureRunner()
