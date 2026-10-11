import pytest
import sys
sys.path.insert(0, "src")
from decimal import Decimal
from foundry.models import (
    ModelSpec, HardwareProfile, WorkloadProfile, TokenDistribution,
    DeploymentCandidate, BackendType, PrecisionType, HardwareVendor,
    PriceProvenance, CapacityType, PerformanceEvidence, EvidenceType,
    EconomicEvaluation, CostAssumptions, DemandAssumptions,
    ServiceRecommendation, RejectedReason, Job, JobStatus,
    RankBy,
)
from foundry.intake import ModelValidator, WorkloadValidator
from foundry.hardware import check_vram_fit, check_precision_support, lookup_gpu
from foundry.candidates import FixtureCandidateProvider, ConservativeCandidateProvider
from foundry.economics import evaluate
from foundry.qualification import QualificationEngine
from foundry.deployment import generate_manifest, manifest_to_yaml
from foundry.reports import generate_markdown_report
from foundry.schemas import export_performance_profile, validate_against_schema, TESSERACT_PERFORMANCE_PROFILE_V1

class TestModelValidation:
    def test_valid_model(self):
        m = ModelSpec(model_id="test/model", model_revision="v1.0")
        hw = HardwareProfile(hardware_id="test-gpu", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0)
        v = ModelValidator()
        r = v.validate(m, [hw])
        assert r.valid

    def test_missing_model_id(self):
        m = ModelSpec(model_id="", model_revision="v1.0")
        hw = HardwareProfile(hardware_id="test-gpu", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0)
        v = ModelValidator()
        r = v.validate(m, [hw])
        assert not r.valid
        assert any("model_id" in e for e in r.errors)

    def test_missing_revision(self):
        m = ModelSpec(model_id="test/model", model_revision="")
        hw = HardwareProfile(hardware_id="test-gpu", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0)
        v = ModelValidator()
        r = v.validate(m, [hw])
        assert not r.valid
        assert any("model_revision" in e for e in r.errors)

    def test_negative_vram(self):
        with pytest.raises(Exception):
            HardwareProfile(hardware_id="test-gpu", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=-1.0)

    def test_no_hardware(self):
        m = ModelSpec(model_id="test/model", model_revision="v1.0")
        v = ModelValidator()
        r = v.validate(m, [])
        assert not r.valid

class TestWorkloadValidation:
    def test_valid_workload(self):
        w = WorkloadProfile(
            workload_id="test",
            request_rate_rps=5.0,
            concurrency=8,
            input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
            output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
        )
        v = WorkloadValidator()
        r = v.validate(w)
        assert r.valid

    def test_negative_rate(self):
        with pytest.raises(Exception):
            WorkloadProfile(
                workload_id="test",
                request_rate_rps=-1.0,
                concurrency=8,
                input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
                output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
            )

    def test_zero_concurrency(self):
        with pytest.raises(Exception):
            WorkloadProfile(
                workload_id="test",
                request_rate_rps=5.0,
                concurrency=0,
                input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
                output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
            )

class TestHardware:
    def test_lookup_gpu(self):
        spec = lookup_gpu("H100")
        assert spec is not None
        assert spec["vram"] == 80

    def test_lookup_gpu_insensitive(self):
        spec = lookup_gpu("a100")
        assert spec is not None
        assert spec["vram"] == 80

    def test_lookup_unknown(self):
        spec = lookup_gpu("nonexistent-gpu")
        assert spec is None

    def test_vram_fit(self):
        ok, needed = check_vram_fit(600_000_000, 16, 4096, 80.0)
        assert ok

    def test_vram_fit_large_model(self):
        ok, needed = check_vram_fit(141_000_000_000, 16, 65536, 24.0)
        assert not ok

class TestCandidates:
    def test_fixture_provider_generates_candidates(self):
        m = ModelSpec(model_id="test/model", model_revision="v1.0", parameter_count=600_000_000, context_length=4096)
        hw = [HardwareProfile(hardware_id="h100-1", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0, device_count=4)]
        w = WorkloadProfile(
            workload_id="test",
            request_rate_rps=5.0,
            concurrency=8,
            input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
            output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
        )
        p = FixtureCandidateProvider()
        candidates = p.generate_candidates(m, hw, w)
        assert len(candidates) > 0
        for c in candidates:
            assert c.candidate_id
            assert c.model_id == "test/model"

    def test_conservative_provider(self):
        m = ModelSpec(model_id="test/model", model_revision="v1.0", parameter_count=600_000_000, context_length=4096)
        hw = [HardwareProfile(hardware_id="h100-1", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0)]
        w = WorkloadProfile(
            workload_id="test", request_rate_rps=1.0, concurrency=2,
            input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
            output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
        )
        p = ConservativeCandidateProvider()
        candidates = p.generate_candidates(m, hw, w)
        assert len(candidates) > 0

class TestEconomics:
    def test_evaluate_basic(self):
        from foundry.models import HardwareProfile, HardwareVendor
        hw = HardwareProfile(hardware_id="h100-1", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0)
        c = DeploymentCandidate(
            candidate_id="test-cand",
            model_id="test/model",
            hardware_id="h100-1",
            serving_engine=BackendType.vllm,
            gpu_count=1,
            expected_vram_gb=40.0,
        )
        w = WorkloadProfile(
            workload_id="test", request_rate_rps=5.0, concurrency=8,
            input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
            output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
        )
        econ = evaluate(c, w, hw)
        assert econ.total_cost_monthly > 0
        assert econ.estimated_revenue_monthly >= 0
        assert isinstance(econ.margin_pct, float)

    def test_evaluate_negative_margin(self):
        from foundry.models import HardwareProfile, HardwareVendor
        hw = HardwareProfile(hardware_id="h100-1", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0, device_count=8)
        c = DeploymentCandidate(
            candidate_id="test-cand",
            model_id="test/model",
            hardware_id="h100-1",
            serving_engine=BackendType.vllm,
            gpu_count=8,
            expected_vram_gb=600.0,
        )
        w = WorkloadProfile(
            workload_id="test", request_rate_rps=0.1, concurrency=1,
            input_tokens=TokenDistribution(mean=100, p50=100, p95=200, p99=300, max=500),
            output_tokens=TokenDistribution(mean=50, p50=50, p95=100, p99=200, max=300),
        )
        econ = evaluate(c, w, hw, input_price_per_million=Decimal("0.05"), output_price_per_million=Decimal("0.15"))
        assert econ.margin_pct < 0

    def test_cost_per_token_calculation(self):
        from foundry.models import HardwareProfile, HardwareVendor
        hw = HardwareProfile(hardware_id="h100-1", vendor=HardwareVendor.nvidia, gpu_model="H100", vram_per_gpu_gb=80.0)
        c = DeploymentCandidate(
            candidate_id="test-cand", model_id="test/model", hardware_id="h100-1",
            serving_engine=BackendType.vllm, gpu_count=1, expected_vram_gb=40.0,
        )
        w = WorkloadProfile(
            workload_id="test", request_rate_rps=10.0, concurrency=16,
            input_tokens=TokenDistribution(mean=1024, p50=512, p95=4096, p99=8192, max=16384),
            output_tokens=TokenDistribution(mean=512, p50=256, p95=2048, p99=4096, max=8192),
        )
        econ = evaluate(c, w, hw)
        assert float(econ.cost_per_million_input_tokens) > 0
        assert float(econ.cost_per_million_output_tokens) > 0

class TestQualification:
    def test_qualify_passes(self):
        c = DeploymentCandidate(
            candidate_id="test-cand", model_id="test/model", hardware_id="h100-1",
            serving_engine=BackendType.vllm, gpu_count=1, expected_vram_gb=40.0,
        )
        w = WorkloadProfile(
            workload_id="test", request_rate_rps=1.0, concurrency=2,
            input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
            output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
        )
        q = QualificationEngine()
        hw = HardwareProfile(hardware_id='h100-1', vendor=HardwareVendor.nvidia, gpu_model='H100', vram_per_gpu_gb=80.0)
        ok, reasons = q.qualify_candidate(c, hw, w)
        assert ok

    def test_recommend_no_feasible(self):
        q = QualificationEngine()
        rec = q.recommend([], {}, {}, WorkloadProfile(
            workload_id="test", request_rate_rps=1.0, concurrency=1,
            input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192),
            output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096),
        ), {}, RankBy.lowest_cost, 50.0)
        assert rec.recommended_candidate_id == "NO_FEASIBLE_DEPLOYMENT"

class TestDeployment:
    def test_generate_manifest(self):
        c = DeploymentCandidate(
            candidate_id="test-cand", model_id="test/model", hardware_id="h100-1",
            serving_engine=BackendType.vllm, gpu_count=1, expected_vram_gb=40.0, max_batch_size=32, max_context_length=4096,
        )
        m = generate_manifest(c)
        assert m["spec"]["engine"]["name"] == "vllm"
        assert m["spec"]["hardware"]["gpuCount"] == 1
        assert "reviewRequired" in str(m["spec"])

    def test_manifest_to_yaml(self):
        c = DeploymentCandidate(
            candidate_id="test-cand", model_id="test/model", hardware_id="h100-1",
            serving_engine=BackendType.vllm, gpu_count=1, expected_vram_gb=40.0, max_batch_size=32, max_context_length=4096,
        )
        m = generate_manifest(c)
        y = manifest_to_yaml(m)
        assert "vllm" in y

class TestSchemas:
    def test_export_performance_profile(self):
        p = export_performance_profile(
            candidate_id="test-cand",
            model_id="test/model",
            model_revision="v1.0",
            hardware_id="h100-1",
            hardware_gpu_model="H100",
            hardware_gpu_count=1,
            hardware_vram_gb=80.0,
            engine="vllm",
            engine_version="0.6.0",
            evidence_type="synthetic",
            workload_id="test-wl", workload_concurrency=8,
            workload_input_token_mean=512.0, workload_output_token_mean=256.0,
            ttft_p95_ms=300.0, itl_p95_ms=25.0,
            completed_requests_per_second=2.0,
        )
        assert p["schema_version"] == "tesseract.performance-profile.v1"
        assert p["model_id"] == "test/model"

    def test_validate_performance_profile(self):
        p = export_performance_profile(
            candidate_id="test-cand", model_id="test/model", model_revision="v1.0",
            hardware_id="h100-1", hardware_gpu_model="H100",
            hardware_gpu_count=1, hardware_vram_gb=80.0,
            engine="vllm", engine_version="0.6.0",
            evidence_type="synthetic",
            workload_id="test-wl", workload_concurrency=8,
            workload_input_token_mean=512.0, workload_output_token_mean=256.0,
        )
        errors = validate_against_schema(p, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert len(errors) == 0

    def test_validate_invalid_profile(self):
        errors = validate_against_schema({"schema_version": "wrong"}, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert len(errors) > 0

class TestReports:
    def test_markdown_report_no_feasible(self):
        rec = ServiceRecommendation(
            recommendation_id="rec-test",
            recommended_candidate_id="NO_FEASIBLE_DEPLOYMENT",
            rejected_candidates=[RejectedReason(candidate_id="cand-1", reason="VRAM exceeded")],
        )
        report = generate_markdown_report(rec, {})
        assert "NO FEASIBLE SERVICE" in report
        assert "VRAM exceeded" in report

    def test_markdown_report_normal(self):
        rec = ServiceRecommendation(
            recommendation_id="rec-test",
            recommended_candidate_id="cand-best",
            feasible_candidates=["cand-best", "cand-alt"],
        )
        report = generate_markdown_report(rec, {})
        assert "RECOMMENDATION" in report

class TestJobs:
    @pytest.fixture(autouse=True)
    def setup_db(self):
        from foundry.database import Base, create_engine
        self._engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self._engine)
        yield

    def test_create_job(self):
        from foundry.database import get_session, Base
        from foundry.jobs import JobRunner
        import foundry.database as db
        original_get_engine = db.get_engine
        original_get_session = db.get_session
        try:
            db.get_engine = lambda: self._engine
            from sqlalchemy.orm import sessionmaker
            Session = sessionmaker(bind=self._engine)
            db.get_session = lambda: Session()
            jr = JobRunner()
            job = jr.create_job("test", {"key": "value"})
            assert job.job_id
            assert job.job_type == "test"
            assert job.status == JobStatus.pending
        finally:
            db.get_engine = original_get_engine
            db.get_session = original_get_session
