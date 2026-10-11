from foundry.models import TokenDistribution, WorkloadProfile, DeploymentCandidate, BackendType
from foundry.economics import evaluate
w = WorkloadProfile(workload_id='test', request_rate_rps=5.0, concurrency=8, input_tokens=TokenDistribution(mean=512, p50=384, p95=2048, p99=4096, max=8192), output_tokens=TokenDistribution(mean=256, p50=128, p95=1024, p99=2048, max=4096))
c = DeploymentCandidate(candidate_id='test', model_id='test/model', hardware_id='l4-1', serving_engine=BackendType.vllm, gpu_count=1, expected_vram_gb=24.0, max_batch_size=32, max_context_length=4096)
econ = evaluate(c, w)
print('Total cost:', float(econ.total_cost_monthly))
print('Formatted:', f'{float(econ.total_cost_monthly):,.2f}')
