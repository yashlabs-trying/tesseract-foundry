import pytest
import sys
sys.path.insert(0, "src")
from foundry.schemas import (
    export_performance_profile, validate_against_schema,
    TESSERACT_PERFORMANCE_PROFILE_V1,
)

class TestPerformanceProfileSchema:
    def test_valid_profile_passes(self):
        p = export_performance_profile(
            candidate_id="test",
            model_id="org/model",
            model_revision="abc123",
            hardware_id="gpu-1",
            hardware_gpu_model="H100",
            hardware_gpu_count=1,
            hardware_vram_gb=80.0,
            engine="vllm",
            engine_version="0.6.0",
            evidence_type="synthetic",
            workload_id="wl-1",
            workload_concurrency=8,
            workload_input_token_mean=512.0,
            workload_output_token_mean=256.0,
        )
        errors = validate_against_schema(p, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert len(errors) == 0

    def test_missing_required_field(self):
        d = {"schema_version": "tesseract.performance-profile.v1"}
        errors = validate_against_schema(d, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert len(errors) > 0
        assert any("model_id" in e for e in errors)

    def test_wrong_schema_version(self):
        p = export_performance_profile(
            candidate_id="test", model_id="org/model", model_revision="abc123",
            hardware_id="gpu-1", hardware_gpu_model="H100",
            hardware_gpu_count=1, hardware_vram_gb=80.0,
            engine="vllm", engine_version="0.6.0",
            evidence_type="synthetic",
            workload_id="wl-1", workload_concurrency=8,
            workload_input_token_mean=512.0, workload_output_token_mean=256.0,
        )
        p["schema_version"] = "wrong"
        errors = validate_against_schema(p, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert any("wrong" in e for e in errors)

    def test_invalid_evidence_type(self):
        p = export_performance_profile(
            candidate_id="test", model_id="org/model", model_revision="abc123",
            hardware_id="gpu-1", hardware_gpu_model="H100",
            hardware_gpu_count=1, hardware_vram_gb=80.0,
            engine="vllm", engine_version="0.6.0",
            evidence_type="fake_type",
            workload_id="wl-1", workload_concurrency=8,
            workload_input_token_mean=512.0, workload_output_token_mean=256.0,
        )
        errors = validate_against_schema(p, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert len(errors) > 0

    def test_negative_gpu_count(self):
        p = export_performance_profile(
            candidate_id="test", model_id="org/model", model_revision="abc123",
            hardware_id="gpu-1", hardware_gpu_model="H100",
            hardware_gpu_count=-1, hardware_vram_gb=80.0,
            engine="vllm", engine_version="0.6.0",
            evidence_type="synthetic",
            workload_id="wl-1", workload_concurrency=8,
            workload_input_token_mean=512.0, workload_output_token_mean=256.0,
        )
        errors = validate_against_schema(p, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert len(errors) > 0