from pathlib import Path
import yaml
from foundry.models import ModelSpec, HardwareProfile, WorkloadProfile
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
    economics_map[c.candidate_id] = evaluate(c, workload)

q = QualificationEngine()
rec = q.recommend(candidates, economics_map, workload, evidence_map)
candidates_by_id = {c.candidate_id: c for c in candidates}

econ = rec.economic_assumptions
print('econ type:', type(econ))
print('total type:', type(econ.total_cost_monthly))
print('total:', econ.total_cost_monthly)
print('float:', float(econ.total_cost_monthly))
try:
    result = f'- **Total Monthly Cost: $ {float(econ.total_cost_monthly):,.2f}**'
    print('RESULT:', repr(result))
except Exception as e:
    print('ERROR:', e)
