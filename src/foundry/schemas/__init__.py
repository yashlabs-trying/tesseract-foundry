import json
from pathlib import Path
from typing import Any

TESSERACT_PERFORMANCE_PROFILE_V1 = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "const": "tesseract.performance-profile.v1"},
        "profile_id": {"type": "string"},
        "model_id": {"type": "string"},
        "model_revision": {"type": "string"},
        "hardware_id": {"type": "string"},
        "engine": {"type": "string"},
        "engine_version": {"type": "string"},
        "candidate_id": {"type": "string"},
        "evidence_type": {"type": "string", "enum": ["synthetic", "estimated", "measured", "validated"]},
        "workload_id": {"type": "string"},
        "gpu_count": {"type": "integer", "minimum": 1},
        "latency_ms": {
            "type": "object",
            "properties": {
                "ttft_p50": {"type": "number"},
                "ttft_p95": {"type": "number"},
                "ttft_p99": {"type": "number"},
                "itl_p50": {"type": "number"},
                "itl_p95": {"type": "number"},
                "request_latency_p50": {"type": "number"},
                "request_latency_p95": {"type": "number"},
            },
        },
        "throughput": {
            "type": "object",
            "properties": {
                "completed_requests_per_second": {"type": "number"},
                "input_tokens_per_second": {"type": "number"},
                "output_tokens_per_second": {"type": "number"},
            },
        },
        "error_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "sample_count": {"type": "integer", "minimum": 0},
        "source": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["test_fixture", "benchmark", "profiler", "external"]},
                "artifact_id": {"type": "string"},
                "collection_timestamp": {"type": "string", "format": "date-time"},
            },
            "required": ["type", "artifact_id"],
        },
    },
    "required": [
        "schema_version", "profile_id", "model_id", "model_revision",
        "hardware_id", "engine", "evidence_type", "source",
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

def export_performance_profile(candidate_id: str, model_id: str, model_revision: str, hardware_id: str, engine: str, evidence_type: str, workload_id: str, gpu_count: int, latency: dict | None = None, throughput: dict | None = None, error_rate: float = 0.0, sample_count: int = 0, source_type: str = "test_fixture", artifact_id: str = "") -> dict:
    import uuid
    profile = {
        "schema_version": "tesseract.performance-profile.v1",
        "profile_id": f"profile-{uuid.uuid4().hex[:8]}",
        "model_id": model_id,
        "model_revision": model_revision,
        "hardware_id": hardware_id,
        "engine": engine,
        "engine_version": "latest",
        "candidate_id": candidate_id,
        "evidence_type": evidence_type,
        "workload_id": workload_id,
        "gpu_count": gpu_count,
        "latency_ms": latency or {},
        "throughput": throughput or {},
        "error_rate": error_rate,
        "sample_count": sample_count,
        "source": {
            "type": source_type,
            "artifact_id": artifact_id,
            "collection_timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        },
    }
    return profile
