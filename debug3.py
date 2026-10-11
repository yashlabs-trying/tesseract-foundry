from pathlib import Path
import yaml
from foundry.models import ModelSpec, HardwareProfile, WorkloadProfile, TokenDistribution
from foundry.candidates import FixtureCandidateProvider
from foundry.benchmarking import FixtureRunner
from foundry.economics import evaluate
from foundry.qualification import QualificationEngine
from foundry.reports import generate_markdown_report
from decimal import Decimal

data = yaml.safe_load(open('fixtures/scenario-1-single-gpu.yaml'))
model = ModelSpec(**data['model'])
hardware = [HardwareProfile(**h) for h in data['hardware']]
workload = WorkloadProfile(**data['workload'])

provider = FixtureCandidateProvider()
candidates = provider.generate_candidates(model, hardware, workload)
runner = FixtureRunner()
evidence_map = {}
economics_map = {}
for c in candidates:
    evidence_map[c.candidate_id] = runner.run(c, workload)
    inp = Decimal(str(data.get('input_price_per_million', 0.15)))
    outp = Decimal(str(data.get('output_price_per_million', 0.60)))
    economics_map[c.candidate_id] = evaluate(c, workload, inp, outp)

q = QualificationEngine()
rec = q.recommend(candidates, economics_map, workload, evidence_map, min_margin=20.0)

print('rec.economic_assumptions:', rec.economic_assumptions)
print('type:', type(rec.economic_assumptions))
if rec.economic_assumptions:
    print('total_cost:', repr(rec.economic_assumptions.total_cost_monthly))
    print('float:', float(rec.economic_assumptions.total_cost_monthly))
    print('formatted:', f'\$ {float(rec.economic_assumptions.total_cost_monthly):,.2f}')
