from ..models import DeploymentCandidate, PerformanceEvidence, BackendType, PrecisionType
from typing import Optional
import json
import os

class DynamoAdapter:
    def __init__(self, api_url: Optional[str] = None, namespace: str = "default"):
        self.api_url = api_url
        self.namespace = namespace
        self._available = api_url is not None

    def is_available(self) -> bool:
        return self._available

    def generate_dgdr(self, candidate: DeploymentCandidate) -> dict:
        return {
            "apiVersion": "nvidia.com/v1",
            "kind": "DynamoGraphDeploymentRequest",
            "metadata": {
                "name": f"foundry-{candidate.candidate_id[:16]}",
                "namespace": self.namespace,
            },
            "spec": {
                "model": {"id": candidate.model_id},
                "hardware": {"gpuCount": candidate.gpu_count, "tensorParallelism": candidate.tensor_parallelism},
                "engine": {"name": candidate.serving_engine.value, "version": candidate.engine_version},
                "autoApply": False,
            },
        }

    def submit_dgdr(self, candidate: DeploymentCandidate) -> Optional[str]:
        if not self._available:
            return None
        dgdr = self.generate_dgdr(candidate)
        return json.dumps(dgdr)

    def extract_candidates(self, raw: dict) -> list[DeploymentCandidate]:
        return []

class AIPerfAdapter:
    def __init__(self, binary_path: Optional[str] = None):
        self.binary_path = binary_path
        self._available = binary_path is not None and os.path.isfile(binary_path) if binary_path else False

    def is_available(self) -> bool:
        return self._available

    def build_command(self, candidate: DeploymentCandidate, model: str, concurrency: int = 1, input_len: int = 512, output_len: int = 256) -> list[str]:
        return [
            self.binary_path, "--model", model,
            "--backend", candidate.serving_engine.value,
            "--concurrency", str(concurrency),
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            "--num-prompts", "100",
        ] if self._available else []
