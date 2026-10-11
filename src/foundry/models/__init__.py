from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator

class EvidenceType(str, Enum):
    synthetic = "synthetic"
    estimated = "estimated"
    measured = "measured"
    validated = "validated"

class BackendType(str, Enum):
    vllm = "vllm"
    sglang = "sglang"
    tensorrt_llm = "tensorrt_llm"
    dynamo = "dynamo"
    default = "default"

class WeightDtype(str, Enum):
    fp16 = "fp16"
    fp32 = "fp32"
    bf16 = "bf16"
    int8 = "int8"
    int4 = "int4"

class PrecisionType(str, Enum):
    fp16 = "fp16"
    fp32 = "fp32"
    bf16 = "bf16"
    fp8 = "fp8"
    int4 = "int4"
    int8 = "int8"

class HardwareVendor(str, Enum):
    nvidia = "nvidia"
    amd = "amd"
    apple = "apple"
    intel = "intel"

class CapacityType(str, Enum):
    owned = "owned"
    reserved = "reserved"
    on_demand = "on_demand"
    spot = "spot"

class PriceProvenance(str, Enum):
    list_price = "list_price"
    negotiated = "negotiated"
    estimated = "estimated"
    published = "published"

class CostBasis(str, Enum):
    accounting_allocation = "accounting_allocation"
    avoidable_cash = "avoidable_cash"
    opportunity_cost = "opportunity_cost"

class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

class RankBy(str, Enum):
    lowest_cost = "lowest_cost"
    highest_margin = "highest_margin"
    lowest_gpu_count = "lowest_gpu_count"
    best_throughput = "best_throughput"
    most_conservative = "most_conservative"

# --- ModelSpec ---
class ModelSpec(BaseModel):
    model_id: str = Field(..., description="Hugging Face style model identifier")
    model_revision: str = Field(..., description="Immutable revision hash or tag")
    source: str = Field("huggingface", description="Source registry")
    architecture: Optional[str] = Field(None, description="Model architecture family")
    parameter_count: Optional[int] = Field(None, ge=1)
    context_length: Optional[int] = Field(None, ge=1)
    weight_dtype: Optional[WeightDtype] = None
    weight_format: Optional[str] = None
    license_info: Optional[str] = Field(None, alias="license_information")
    access_requirements: Optional[str] = None
    supported_backends: list[BackendType] = Field(default_factory=lambda: [BackendType.default])
    status: str = "unknown"

    class Config:
        populate_by_name = True

# --- HardwareProfile ---
class HardwareProfile(BaseModel):
    hardware_id: str
    vendor: HardwareVendor
    gpu_model: str
    gpu_sku: Optional[str] = None
    device_count: int = Field(default=1, ge=1)
    vram_per_gpu_gb: float = Field(..., gt=0)
    memory_bandwidth_gbps: Optional[float] = None
    interconnect: Optional[str] = None
    topology: Optional[str] = None
    supported_precision: list[PrecisionType] = Field(default_factory=list)
    available_features: list[str] = Field(default_factory=list)
    cost_per_gpu_hour_usd: Decimal = Field(default=Decimal("0"), ge=0)
    price_provenance: PriceProvenance = PriceProvenance.list_price
    capacity_type: CapacityType = CapacityType.owned

# --- WorkloadProfile ---
class TokenDistribution(BaseModel):
    mean: float = Field(..., gt=0)
    p50: float = Field(..., gt=0)
    p95: float = Field(..., gt=0)
    p99: float = Field(..., gt=0)
    max: float = Field(..., gt=0)

class WorkloadProfile(BaseModel):
    workload_id: str
    request_rate_rps: float = Field(..., gt=0)
    concurrency: int = Field(default=1, ge=1)
    input_tokens: TokenDistribution
    output_tokens: TokenDistribution
    burst_multiplier: float = Field(default=2.0, ge=1.0)
    peak_hours_per_day: float = Field(default=8, ge=0, le=24)
    idle_hours_per_day: float = Field(default=8, ge=0, le=24)
    warm_latency_target_ms: Optional[float] = Field(None, gt=0)
    cold_activation_seconds: Optional[float] = Field(None, gt=0)
    reliability_objective: float = Field(default=0.99, ge=0, le=1)

# --- DeploymentCandidate ---
class DeploymentCandidate(BaseModel):
    candidate_id: str
    model_id: str
    hardware_id: str
    serving_engine: BackendType
    engine_version: str = "latest"
    precision: PrecisionType = PrecisionType.fp16
    quantization_format: Optional[str] = None
    tensor_parallelism: int = Field(default=1, ge=1)
    pipeline_parallelism: int = Field(default=1, ge=1)
    prefill_decode_topology: Optional[str] = None
    gpu_count: int = Field(default=1, ge=1)
    replica_count: int = Field(default=1, ge=1)
    max_batch_size: int = Field(default=32, ge=1)
    max_context_length: int = Field(default=4096, ge=1)
    model_revision: str = ""
    model_source: str = "huggingface"
    expected_vram_gb: float = Field(default=0, ge=0)
    deployment_manifest_ref: Optional[str] = None
    estimated: bool = True
    confidence: str = "low"

# --- PerformanceEvidence ---
class MetricAggregate(BaseModel):
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unit: str = "ms"

class ThroughputMetrics(BaseModel):
    completed_requests_per_second: Optional[float] = None
    input_tokens_per_second: Optional[float] = None
    output_tokens_per_second: Optional[float] = None

class PerformanceEvidence(BaseModel):
    evidence_id: str
    candidate_id: str
    evidence_type: EvidenceType
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    benchmark_tool: str = "fixture"
    benchmark_version: str = "1.0"
    model_revision: str = ""
    container_image_digest: Optional[str] = None
    gpu_architecture: Optional[str] = None
    benchmark_config: dict = Field(default_factory=dict)
    raw_artifact_ref: Optional[str] = None
    ttft_ms: Optional[MetricAggregate] = None
    itl_ms: Optional[MetricAggregate] = None
    request_latency_ms: Optional[MetricAggregate] = None
    throughput: Optional[ThroughputMetrics] = None
    error_rate: float = 0.0
    sample_count: int = 0
    valid: bool = True
    confidence_limitations: Optional[str] = None

# --- EconomicEvaluation ---
class CostAssumptions(BaseModel):
    cost_basis: CostBasis = CostBasis.avoidable_cash
    gpu_hourly_cost: Decimal = Field(default=Decimal("0"), ge=0)
    attributable_overhead_pct: Decimal = Field(default=Decimal("5"), ge=0)
    energy_cost_per_kwh: Decimal = Field(default=Decimal("0.10"), ge=0)
    estimated_power_draw_watts: float = Field(default=700, gt=0)

class DemandAssumptions(BaseModel):
    monthly_request_volume: int = Field(default=0, ge=0)
    avg_input_tokens_per_request: float = Field(default=512, gt=0)
    avg_output_tokens_per_request: float = Field(default=256, gt=0)
    peak_demand_multiplier: float = Field(default=2.0, ge=1.0)

class SensitivityAnalysis(BaseModel):
    demand_range: list[float] = Field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])
    gpu_cost_range: list[float] = Field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])

class EconomicEvaluation(BaseModel):
    candidate_id: str
    scenario_id: str
    cost_assumptions: CostAssumptions
    demand_assumptions: DemandAssumptions
    allocated_gpu_cost_monthly: Decimal = Field(default=Decimal("0"))
    attributable_variable_cost_monthly: Decimal = Field(default=Decimal("0"))
    total_cost_monthly: Decimal = Field(default=Decimal("0"))
    estimated_revenue_monthly: Decimal = Field(default=Decimal("0"))
    contribution_margin: Decimal = Field(default=Decimal("0"))
    margin_pct: float = 0.0
    break_even_requests_per_month: Optional[float] = None
    predicted_utilization_pct: float = 0.0
    cost_per_million_input_tokens: Decimal = Field(default=Decimal("0"))
    cost_per_million_output_tokens: Decimal = Field(default=Decimal("0"))
    cost_per_request: Decimal = Field(default=Decimal("0"))
    revenue_per_request: Decimal = Field(default=Decimal("0"))
    sensitivity: SensitivityAnalysis = Field(default_factory=SensitivityAnalysis)
    sensitivity_results: dict[str, list[Decimal]] = Field(default_factory=dict)
    uncertainty_notes: Optional[str] = None
    status: str = "estimated"
    config_version: str = "2.0"

# --- ServiceRecommendation ---
class RejectedReason(BaseModel):
    candidate_id: str
    reason: str

class ServiceRecommendation(BaseModel):
    recommendation_id: str
    recommended_candidate_id: str
    feasible_candidates: list[str] = Field(default_factory=list)
    rejected_candidates: list[RejectedReason] = Field(default_factory=list)
    expected_performance: Optional[PerformanceEvidence] = None
    economic_assumptions: Optional[EconomicEvaluation] = None
    proposed_pricing: Optional[dict] = None
    deployment_artifact_ref: Optional[str] = None
    approval_status: str = "pending_review"
    created_at: datetime = Field(default_factory=datetime.utcnow)

# --- Job ---
class Job(BaseModel):
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.pending
    params: dict = Field(default_factory=dict)
    result_ref: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# --- Scenario ---
class Scenario(BaseModel):
    scenario_id: str
    model: ModelSpec
    hardware: list[HardwareProfile]
    workload: WorkloadProfile
    rank_by: RankBy = RankBy.lowest_cost
    min_contribution_margin_pct: float = Field(default=20.0, ge=0, le=100)
    max_gpu_budget: Optional[int] = None
    output_price_per_million_input_tokens: Optional[Decimal] = None
    output_price_per_million_output_tokens: Optional[Decimal] = None
