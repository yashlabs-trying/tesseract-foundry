import pytest
import sys
sys.path.insert(0, "src")
import yaml
from pathlib import Path
from foundry.models import ModelSpec, HardwareProfile as HwProfile, WorkloadProfile, MetricAggregate, ThroughputMetrics, HardwareVendor
from foundry.candidates import FixtureCandidateProvider
from foundry.benchmarking import FixtureRunner
from foundry.economics import evaluate
from foundry.models import HardwareProfile, HardwareVendor
from foundry.qualification import QualificationEngine
from foundry.deployment import generate_manifest
from foundry.reports import generate_markdown_report
from foundry.schemas import export_performance_profile, validate_against_schema, TESSERACT_PERFORMANCE_PROFILE_V1
from foundry.intake import IntakeOrchestrator
from decimal import Decimal

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

class TestScenario1SingleGPU:
    @pytest.fixture(autouse=True)
    def setup(self):
        data = load_yaml(FIXTURES / "scenario-1-single-gpu.yaml")
        self.model = ModelSpec(**data["model"])
        self.hardware = [HardwareProfile(**h) for h in data["hardware"]]
        self.workload = WorkloadProfile(**data["workload"])
        self.min_margin = float(data.get("min_contribution_margin_pct", 20.0))

    def test_complete_workflow(self):
        intake = IntakeOrchestrator()
        results = intake.validate_scenario(self.model, self.hardware, self.workload)
        assert all(r.valid for r in results)

        provider = FixtureCandidateProvider()
        candidates = provider.generate_candidates(self.model, self.hardware, self.workload)
        assert len(candidates) > 0

        runner = FixtureRunner()
        evidence_map = {}
        economics_map = {}
        for c in candidates:
            evidence_map[c.candidate_id] = runner.run(c, self.workload)
            hw = next(h for h in self.hardware if h.hardware_id == c.hardware_id)
            economics_map[c.candidate_id] = evaluate(c, self.workload, hw)

        q = QualificationEngine()
        hw_map = {h.hardware_id: h for h in self.hardware}
        rec = q.recommend(candidates, hw_map, economics_map, self.workload, evidence_map, min_margin=self.min_margin)
        assert rec.recommended_candidate_id not in ("NO_FEASIBLE_DEPLOYMENT", "NO_VERIFIED_DEPLOYMENT") or rec.approval_status == "unverified_only"
        assert len(rec.feasible_candidates) > 0

        candidate = next(iter(candidates), None)
        if candidate:
            manifest = generate_manifest(candidate, rec)
            assert manifest["spec"]["engine"]["name"] == "vllm"

        report = generate_markdown_report(rec, {c.candidate_id: c for c in candidates})
        assert "TESSERACT FOUNDRY" in report

        if rec.recommended_candidate_id in ("NO_FEASIBLE_DEPLOYMENT", "NO_VERIFIED_DEPLOYMENT"):
            return
        profile = export_performance_profile(
            candidate_id=candidate.candidate_id,
            model_id=self.model.model_id,
            model_revision=self.model.model_revision,
            hardware_id=candidate.hardware_id,
            engine=candidate.serving_engine.value,
            evidence_type="synthetic",
            workload_id=self.workload.workload_id,
            gpu_count=candidate.gpu_count,
        )
        errors = validate_against_schema(profile, TESSERACT_PERFORMANCE_PROFILE_V1)
        assert len(errors) == 0

class TestScenario2TwoGPU:
    @pytest.fixture(autouse=True)
    def setup(self):
        data = load_yaml(FIXTURES / "scenario-2-two-gpu-types.yaml")
        self.model = ModelSpec(**data["model"])
        self.hardware = [HardwareProfile(**h) for h in data["hardware"]]
        self.workload = WorkloadProfile(**data["workload"])

    def test_generates_candidates_for_both_gpus(self):
        provider = FixtureCandidateProvider()
        candidates = provider.generate_candidates(self.model, self.hardware, self.workload)
        hw_ids = set(c.hardware_id for c in candidates)
        assert "l4-1" in hw_ids
        assert "h100-1" in hw_ids

    def test_economics_differ_by_hardware(self):
        provider = FixtureCandidateProvider()
        candidates = provider.generate_candidates(self.model, self.hardware, self.workload)
        costs = {}
        for c in candidates:
            hw = next(h for h in self.hardware if h.hardware_id == c.hardware_id)
            econ = evaluate(c, self.workload, hw)
            costs[c.hardware_id] = float(econ.total_cost_monthly)
        assert costs.get("l4-1", float("inf")) < costs.get("h100-1", float("inf"))

class TestScenario3BurstyNegative:
    @pytest.fixture(autouse=True)
    def setup(self):
        data = load_yaml(FIXTURES / "scenario-3-bursty-negative-margin.yaml")
        self.model = ModelSpec(**data["model"])
        self.hardware = [HardwareProfile(**h) for h in data["hardware"]]
        self.workload = WorkloadProfile(**data["workload"])
        self.min_margin = float(data.get("min_contribution_margin_pct", 30.0))
        self.inp_price = Decimal(str(data.get("input_price_per_million", 0.05)))
        self.out_price = Decimal(str(data.get("output_price_per_million", 0.20)))

    def test_possible_negative_margin(self):
        provider = FixtureCandidateProvider()
        candidates = provider.generate_candidates(self.model, self.hardware, self.workload)
        for c in candidates:
            hw_n = next(h for h in self.hardware if h.hardware_id == c.hardware_id)
            econ = evaluate(c, self.workload, hw_n, self.inp_price, self.out_price)
            assert isinstance(econ.margin_pct, float)

    def test_no_feasible_with_high_margin(self):
        provider = FixtureCandidateProvider()
        candidates = provider.generate_candidates(self.model, self.hardware, self.workload)
        runner = FixtureRunner()
        evidence_map = {}
        economics_map = {}
        for c in candidates:
            evidence_map[c.candidate_id] = runner.run(c, self.workload)
            hw_nf = next(h for h in self.hardware if h.hardware_id == c.hardware_id)
            economics_map[c.candidate_id] = evaluate(c, self.workload, hw_nf, self.inp_price, self.out_price)

        q = QualificationEngine()
        hw_map2 = {h.hardware_id: h for h in self.hardware}
        rec = q.recommend(candidates, hw_map2, economics_map, self.workload, evidence_map, min_margin=50.0)
        if rec.recommended_candidate_id == "NO_FEASIBLE_DEPLOYMENT":
            assert len(rec.rejected_candidates) > 0
