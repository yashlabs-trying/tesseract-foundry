import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TESSERACT_PERFORMANCE_PROFILE_V1 = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "const": "tesseract.performance-profile.v1"},
        "profile_id": {"type": "string"},
        "model_id": {"type": "string"},
        "model_revision": {"type": "string"},
        "model_parameter_count": {"type": "integer", "minimum": 1},
        "hardware_id": {"type": "string"},
        "hardware_gpu_model": {"type": "string"},
        "hardware_gpu_count": {"type": "integer", "minimum": 1},
        "hardware_vram_gb": {"type": "number", "minimum": 0},
        "engine": {"type": "string"},
        "engine_version": {"type": "string"},
        "candidate_id": {"type": "string"},
        "evidence_type": {"type": "string", "enum": ["synthetic", "estimated", "measured", "validated"]},
        "benchmark_tool": {"type": "string"},
        "benchmark_version": {"type": "string"},
        "workload_id": {"type": "string"},
        "workload_concurrency": {"type": "integer", "minimum": 1},
        "workload_input_token_mean": {"type": "number", "minimum": 1},
        "workload_output_token_mean": {"type": "number", "minimum": 1},
        "ttft_p50_ms": {"type": "number", "minimum": 0},
        "ttft_p95_ms": {"type": "number", "minimum": 0},
        "ttft_p99_ms": {"type": "number", "minimum": 0},
        "itl_p50_ms": {"type": "number", "minimum": 0},
        "itl_p95_ms": {"type": "number", "minimum": 0},
        "itl_p99_ms": {"type": "number", "minimum": 0},
        "request_latency_p50_ms": {"type": "number", "minimum": 0},
        "request_latency_p95_ms": {"type": "number", "minimum": 0},
        "request_latency_p99_ms": {"type": "number", "minimum": 0},
        "completed_requests_per_second": {"type": "number", "minimum": 0},
        "input_tokens_per_second": {"type": "number", "minimum": 0},
        "output_tokens_per_second": {"type": "number", "minimum": 0},
        "error_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "sample_count": {"type": "integer", "minimum": 0},
        "collection_timestamp": {"type": "string", "format": "date-time"},
        "source_type": {"type": "string", "enum": ["test_fixture", "benchmark", "profiler", "external"]},
        "source_artifact_id": {"type": "string"},
    },
    "required": [
        "schema_version", "profile_id", "model_id", "model_revision",
        "hardware_id", "engine", "evidence_type", "collection_timestamp", "source_type",
    ],
}


def validate_against_schema(data: dict, schema: dict) -> list[str]:
    errors = []
    for required in schema.get("required", []):
        if required not in data:
            errors.append(f"Missing required field: {required}")
    props = schema.get("properties", {})
    for key, value in data.items():
        if key not in props:
            continue
        prop = props[key]
        prop_type = prop.get("type")
        if prop_type == "object" and isinstance(value, dict):
            nested = validate_against_schema(value, prop)
            errors.extend(nested)
        elif prop_type == "integer" and not isinstance(value, int):
            errors.append(f"{key}: expected integer, got {type(value).__name__}")
        elif prop_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"{key}: expected number, got {type(value).__name__}")
        elif prop_type == "string" and not isinstance(value, str):
            errors.append(f"{key}: expected string, got {type(value).__name__}")
        if "minimum" in prop and isinstance(value, (int, float)) and value < prop["minimum"]:
            errors.append(f"{key}: {value} is below minimum {prop['minimum']}")
        if "maximum" in prop and isinstance(value, (int, float)) and value > prop["maximum"]:
            errors.append(f"{key}: {value} is above maximum {prop['maximum']}")
        if "enum" in prop and value not in prop["enum"]:
            errors.append(f"{key}: '{value}' not in allowed values {prop['enum']}")
        if "const" in prop and value != prop["const"]:
            errors.append(f"{key}: expected '{prop['const']}', got '{value}'")
    return errors


def export_performance_profile(
    candidate_id: str,
    model_id: str,
    model_revision: str,
    hardware_id: str,
    hardware_gpu_model: str,
    hardware_gpu_count: int,
    hardware_vram_gb: float,
    engine: str,
    engine_version: str,
    evidence_type: str,
    workload_id: str,
    workload_concurrency: int,
    workload_input_token_mean: float,
    workload_output_token_mean: float,
    ttft_p50_ms: Optional[float] = None,
    ttft_p95_ms: Optional[float] = None,
    ttft_p99_ms: Optional[float] = None,
    itl_p50_ms: Optional[float] = None,
    itl_p95_ms: Optional[float] = None,
    itl_p99_ms: Optional[float] = None,
    request_latency_p50_ms: Optional[float] = None,
    request_latency_p95_ms: Optional[float] = None,
    request_latency_p99_ms: Optional[float] = None,
    completed_requests_per_second: Optional[float] = None,
    input_tokens_per_second: Optional[float] = None,
    output_tokens_per_second: Optional[float] = None,
    error_rate: float = 0.0,
    sample_count: int = 0,
    model_parameter_count: Optional[int] = None,
    benchmark_tool: str = "fixture",
    benchmark_version: str = "1.0",
    source_type: str = "test_fixture",
    source_artifact_id: str = "",
) -> dict:
    import uuid
    profile = {
        "schema_version": "tesseract.performance-profile.v1",
        "profile_id": f"profile-{uuid.uuid4().hex[:8]}",
        "model_id": model_id,
        "model_revision": model_revision,
        "hardware_id": hardware_id,
        "hardware_gpu_model": hardware_gpu_model,
        "hardware_gpu_count": hardware_gpu_count,
        "hardware_vram_gb": hardware_vram_gb,
        "engine": engine,
        "engine_version": engine_version,
        "candidate_id": candidate_id,
        "evidence_type": evidence_type,
        "benchmark_tool": benchmark_tool,
        "benchmark_version": benchmark_version,
        "workload_id": workload_id,
        "workload_concurrency": workload_concurrency,
        "workload_input_token_mean": workload_input_token_mean,
        "workload_output_token_mean": workload_output_token_mean,
        "error_rate": error_rate,
        "sample_count": sample_count,
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "source_artifact_id": source_artifact_id,
    }
    optional_fields = {
        "model_parameter_count": model_parameter_count,
        "ttft_p50_ms": ttft_p50_ms,
        "ttft_p95_ms": ttft_p95_ms,
        "ttft_p99_ms": ttft_p99_ms,
        "itl_p50_ms": itl_p50_ms,
        "itl_p95_ms": itl_p95_ms,
        "itl_p99_ms": itl_p99_ms,
        "request_latency_p50_ms": request_latency_p50_ms,
        "request_latency_p95_ms": request_latency_p95_ms,
        "request_latency_p99_ms": request_latency_p99_ms,
        "completed_requests_per_second": completed_requests_per_second,
        "input_tokens_per_second": input_tokens_per_second,
        "output_tokens_per_second": output_tokens_per_second,
    }
    for k, v in optional_fields.items():
        if v is not None:
            profile[k] = v
    return profile
