from ..models import ModelSpec, HardwareProfile, WorkloadProfile, BackendType, PrecisionType
from typing import Union
from dataclasses import dataclass

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]

class ModelValidator:
    def validate(self, model: ModelSpec, hardware: list[HardwareProfile]) -> ValidationResult:
        errors = []
        warnings = []
        if not model.model_id:
            errors.append("model_id is required")
        if not model.model_revision:
            errors.append("model_revision is required")
        if model.context_length and model.context_length < 1:
            errors.append("context_length must be positive")
        if not hardware:
            errors.append("at least one hardware profile is required")
        for hw in hardware:
            if hw.vram_per_gpu_gb <= 0:
                errors.append(f"hardware {hw.hardware_id}: vram_per_gpu_gb must be positive")
            if hw.device_count < 1:
                errors.append(f"hardware {hw.hardware_id}: device_count must be at least 1")
        if BackendType.default in model.supported_backends:
            warnings.append("model has no explicit backend support declared; using default")
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

class WorkloadValidator:
    def validate(self, workload: WorkloadProfile) -> ValidationResult:
        errors = []
        warnings = []
        if workload.request_rate_rps <= 0:
            errors.append("request_rate_rps must be positive")
        if workload.concurrency < 1:
            errors.append("concurrency must be at least 1")
        for dist_name, dist in [("input_tokens", workload.input_tokens), ("output_tokens", workload.output_tokens)]:
            for field_name in ["mean", "p50", "p95", "p99", "max"]:
                val = getattr(dist, field_name)
                if val <= 0:
                    errors.append(f"{dist_name}.{field_name} must be positive")
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

class IntakeOrchestrator:
    def __init__(self):
        self.model_validator = ModelValidator()
        self.workload_validator = WorkloadValidator()

    def validate_scenario(self, model: ModelSpec, hardware: list[HardwareProfile], workload: WorkloadProfile) -> list[ValidationResult]:
        return [
            self.model_validator.validate(model, hardware),
            self.workload_validator.validate(workload),
        ]
