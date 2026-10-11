from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time
import uuid
import json

from .. import __version__
from ..config import settings
from ..models import (
    ModelSpec, HardwareProfile, WorkloadProfile, Job, JobStatus,
    DeploymentCandidate, PerformanceEvidence, EconomicEvaluation,
    ServiceRecommendation, Scenario, RejectedReason, RankBy,
    EvidenceType,
)
from ..intake import IntakeOrchestrator
from ..candidates import get_provider
from ..benchmarking import get_runner
from ..economics import evaluate
from ..qualification import QualificationEngine
from ..deployment import generate_manifest, manifest_to_yaml
from ..reports import generate_markdown_report
from ..schemas import export_performance_profile, validate_against_schema, TESSERACT_PERFORMANCE_PROFILE_V1
from ..jobs import JobRunner
from ..database import get_session, RecommendationRecord
from datetime import datetime

_start_time = time.time()
intake = IntakeOrchestrator()
qualification = QualificationEngine()
job_runner = JobRunner(max_concurrent=settings.max_job_concurrency)

app = FastAPI(
    title="Tesseract Foundry API",
    version=__version__,
    description="AI deployment intelligence for GPU infrastructure providers",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if settings.api_key_enabled:
        expected = settings.api_key
        if not expected or x_api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return x_api_key

# Health
@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": __version__}

@app.get("/readyz")
def readyz():
    return {"status": "ready"}

# Model validation
@app.post("/v1/models/validate")
def validate_model(model: ModelSpec, hardware: list[HardwareProfile]):
    results = intake.model_validator.validate(model, hardware)
    if not results.valid:
        raise HTTPException(status_code=422, detail={"errors": results.errors, "warnings": results.warnings})
    return {"valid": True, "warnings": results.warnings}

# Hardware profiles
_hardware_store: dict[str, HardwareProfile] = {}

@app.post("/v1/hardware/profiles")
def create_hardware_profile(profile: HardwareProfile, api_key: str = Depends(get_api_key)):
    _hardware_store[profile.hardware_id] = profile
    return {"hardware_id": profile.hardware_id, "status": "created"}

@app.get("/v1/hardware/profiles")
def list_hardware_profiles():
    return list(_hardware_store.values())

@app.get("/v1/hardware/profiles/{hw_id}")
def get_hardware_profile(hw_id: str):
    if hw_id not in _hardware_store:
        raise HTTPException(status_code=404, detail="Hardware profile not found")
    return _hardware_store[hw_id]

# Workload validation
@app.post("/v1/workloads/validate")
def validate_workload(workload: WorkloadProfile):
    results = intake.workload_validator.validate(workload)
    if not results.valid:
        raise HTTPException(status_code=422, detail={"errors": results.errors, "warnings": results.warnings})
    return {"valid": True, "warnings": results.warnings}

# Profiling jobs
@app.post("/v1/profiling/jobs")
def create_profiling_job(model: ModelSpec, hardware: list[HardwareProfile], workload: WorkloadProfile, mode: str = Query("fixture"), api_key: str = Depends(get_api_key)):
    val_results = intake.validate_scenario(model, hardware, workload)
    for vr in val_results:
        if not vr.valid:
            raise HTTPException(status_code=422, detail={"errors": vr.errors, "warnings": vr.warnings})

    provider = get_provider(mode)
    candidates = provider.generate_candidates(model, hardware, workload)
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates could be generated")

    job_params = {
        "model": model.model_dump(),
        "hardware": [h.model_dump() for h in hardware],
        "workload": workload.model_dump(),
        "candidates": [c.model_dump() for c in candidates],
        "mode": mode,
    }
    job = job_runner.create_job("profiling", job_params)
    result_ref = json.dumps([c.candidate_id for c in candidates])
    job_runner.update_status(job.job_id, JobStatus.completed, result_ref=result_ref)
    return {"job_id": job.job_id, "status": "completed", "candidate_count": len(candidates), "candidates": [c.candidate_id for c in candidates]}

@app.get("/v1/profiling/jobs/{job_id}")
def get_profiling_job(job_id: str):
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()

# Benchmark jobs
_benchmark_store: dict[str, PerformanceEvidence] = {}

@app.post("/v1/benchmarks/jobs")
def create_benchmark_job(candidates: list[DeploymentCandidate], workload: WorkloadProfile, mode: str = Query("fixture"), api_key: str = Depends(get_api_key)):
    job = job_runner.create_job("benchmarking", {"candidates": [c.model_dump() for c in candidates], "workload": workload.model_dump(), "mode": mode})
    runner = get_runner(mode)
    for candidate in candidates:
        evidence = runner.run(candidate, workload)
        _benchmark_store[evidence.evidence_id] = evidence
    result_ref = json.dumps(list(_benchmark_store.keys()))
    job_runner.update_status(job.job_id, JobStatus.completed, result_ref=result_ref)
    return {"job_id": job.job_id, "status": "completed", "evidence_count": len(_benchmark_store)}

@app.get("/v1/benchmarks/jobs/{job_id}")
def get_benchmark_job(job_id: str):
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()

# Economics evaluation
@app.post("/v1/economics/evaluate")
def evaluate_economics(candidate: DeploymentCandidate, workload: WorkloadProfile, input_price: Optional[float] = Query(None), output_price: Optional[float] = Query(None), api_key: str = Depends(get_api_key)):
    from decimal import Decimal
    inp = Decimal(str(input_price)) if input_price else None
    outp = Decimal(str(output_price)) if output_price else None
    econ = evaluate(candidate, workload, inp, outp)
    return econ.model_dump()

# Service recommendations
_recommendation_store: dict[str, ServiceRecommendation] = {}
_candidate_store: dict[str, DeploymentCandidate] = {}

@app.post("/v1/services/recommend")
def recommend(scenario: Scenario, evidence: Optional[list[PerformanceEvidence]] = None, api_key: str = Depends(get_api_key)):
    model = scenario.model
    hardware = scenario.hardware
    workload = scenario.workload

    val_results = intake.validate_scenario(model, hardware, workload)
    for vr in val_results:
        if not vr.valid:
            raise HTTPException(status_code=422, detail={"errors": vr.errors, "warnings": vr.warnings})

    provider = get_provider("fixture")
    candidates = provider.generate_candidates(model, hardware, workload)
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates could be generated")

    runner = get_runner("fixture")
    evidence_map: dict[str, PerformanceEvidence] = {}
    if evidence:
        for e in evidence:
            evidence_map[e.candidate_id] = e
    else:
        for c in candidates:
            evidence_map[c.candidate_id] = runner.run(c, workload)

    economics_map: dict[str, EconomicEvaluation] = {}
    for c in candidates:
        inp = scenario.output_price_per_million_input_tokens
        outp = scenario.output_price_per_million_output_tokens
        econ = evaluate(c, workload, inp, outp)
        economics_map[c.candidate_id] = econ

    hardware_map = {h.hardware_id: h for h in scenario.hardware}
    rec = qualification.recommend(candidates, hardware_map, economics_map, workload, evidence_map, scenario.rank_by, scenario.min_contribution_margin_pct)
    _recommendation_store[rec.recommendation_id] = rec
    for c in candidates:
        _candidate_store[c.candidate_id] = c

    return rec.model_dump()

@app.get("/v1/services/recommendations/{rec_id}")
def get_recommendation(rec_id: str):
    if rec_id not in _recommendation_store:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return _recommendation_store[rec_id].model_dump()

@app.get("/v1/services/recommendations/{rec_id}/report")
def get_recommendation_report(rec_id: str, format: str = Query("markdown")):
    if rec_id not in _recommendation_store:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    rec = _recommendation_store[rec_id]
    if format == "json":
        from ..reports import generate_json_report
        return generate_json_report(rec)
    report = generate_markdown_report(rec, _candidate_store)
    return {"report": report}

@app.get("/v1/services/recommendations/{rec_id}/manifest")
def get_recommendation_manifest(rec_id: str):
    if rec_id not in _recommendation_store:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    rec = _recommendation_store[rec_id]
    if rec.recommended_candidate_id == "NO_FEASIBLE_DEPLOYMENT":
        raise HTTPException(status_code=400, detail="No feasible deployment to generate manifest for")
    candidate = _candidate_store.get(rec.recommended_candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    manifest = generate_manifest(candidate, rec)
    return {"manifest": manifest}

# Schema validation
@app.post("/v1/schemas/validate")
def validate_schema(schema_name: str = Query(...), data: dict = None):
    schemas = {
        "tesseract.performance-profile.v1": TESSERACT_PERFORMANCE_PROFILE_V1,
    }
    if schema_name not in schemas:
        raise HTTPException(status_code=404, detail=f"Unknown schema: {schema_name}")
    errors = validate_against_schema(data, schemas[schema_name])
    return {"valid": len(errors) == 0, "errors": errors}

# Metrics
@app.get("/metrics")
def get_metrics():
    return {
        "uptime_seconds": time.time() - _start_time,
        "recommendations": len(_recommendation_store),
        "candidates": len(_candidate_store),
        "hardware_profiles": len(_hardware_store),
        "benchmark_evidence": len(_benchmark_store),
    }
