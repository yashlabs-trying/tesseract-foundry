from ..models import (
    ModelSpec, HardwareProfile, WorkloadProfile, DeploymentCandidate,
    BackendType, PrecisionType,
)
from ..hardware import check_vram_fit, check_backend_compatibility, check_precision_support
from typing import Protocol, Optional
import uuid

class CandidateProvider(Protocol):
    def generate_candidates(self, model: ModelSpec, hardware: list[HardwareProfile], workload: WorkloadProfile) -> list[DeploymentCandidate]: ...

class FixtureCandidateProvider:
    def generate_candidates(self, model: ModelSpec, hardware: list[HardwareProfile], workload: WorkloadProfile) -> list[DeploymentCandidate]:
        candidates = []
        for hw in hardware:
            backend = BackendType.vllm
            ok, _ = check_backend_compatibility(hw, "vllm")
            if not ok:
                continue
            for tp in [1, 2, 4]:
                if tp > hw.device_count:
                    continue
                ok, _ = check_precision_support(hw, PrecisionType.fp16)
                if not ok:
                    continue
                vram_ok, expected = check_vram_fit(
                    model.parameter_count, 16, model.context_length or 4096, hw.vram_per_gpu_gb
                )
                if not vram_ok:
                    expected = hw.vram_per_gpu_gb
                cid = f"candidate-{uuid.uuid4().hex[:8]}"
                candidates.append(DeploymentCandidate(
                    candidate_id=cid,
                    model_id=model.model_id,
                    hardware_id=hw.hardware_id,
                    serving_engine=backend,
                    engine_version="0.6.0",
                    precision=PrecisionType.fp16,
                    tensor_parallelism=tp,
                    pipeline_parallelism=1,
                    gpu_count=tp,
                    replica_count=1,
                    max_batch_size=32,
                    max_context_length=model.context_length or 4096,
                    expected_vram_gb=expected,
                    estimated=True,
                    confidence="low",
                ))
        return candidates

class ConservativeCandidateProvider:
    def generate_candidates(self, model: ModelSpec, hardware: list[HardwareProfile], workload: WorkloadProfile) -> list[DeploymentCandidate]:
        candidates = []
        for hw in hardware:
            for backend in [BackendType.vllm]:
                ok, _ = check_backend_compatibility(hw, backend.value)
                if not ok:
                    continue
                for tp in [1]:
                    if tp > hw.device_count:
                        continue
                    vram_ok, expected = check_vram_fit(
                        model.parameter_count, 16, model.context_length or 4096, hw.vram_per_gpu_gb
                    )
                    cid = f"candidate-{uuid.uuid4().hex[:8]}"
                    candidates.append(DeploymentCandidate(
                        candidate_id=cid,
                        model_id=model.model_id,
                        hardware_id=hw.hardware_id,
                        serving_engine=backend,
                        engine_version="latest",
                        precision=PrecisionType.fp16,
                        tensor_parallelism=tp,
                        pipeline_parallelism=1,
                        gpu_count=tp,
                        replica_count=1,
                        max_batch_size=32,
                        max_context_length=model.context_length or 4096,
                        expected_vram_gb=expected,
                        estimated=True,
                        confidence="medium",
                    ))
        return candidates

def get_provider(mode: str = "fixture") -> CandidateProvider:
    if mode == "conservative":
        return ConservativeCandidateProvider()
    return FixtureCandidateProvider()
