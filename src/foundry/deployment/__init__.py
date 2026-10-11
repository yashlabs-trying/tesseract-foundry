from ..models import DeploymentCandidate, ServiceRecommendation
from typing import Optional
import yaml
import json

def generate_manifest(candidate: DeploymentCandidate, recommendation: Optional[ServiceRecommendation] = None) -> dict:
    manifest = {
        "apiVersion": "v1",
        "kind": "DeploymentManifest",
        "metadata": {
            "name": f"foundry-{candidate.model_id.replace('/', '-')}",
            "labels": {
                "app": "tesseract-foundry",
                "model": candidate.model_id,
                "candidate": candidate.candidate_id,
            },
        },
        "spec": {
            "model": {
                "id": candidate.model_id,
                "revision": candidate.model_revision if hasattr(candidate, 'model_revision') else "latest",
            },
            "hardware": {
                "gpu_count": candidate.gpu_count,
                "tensor_parallelism": candidate.tensor_parallelism,
                "pipeline_parallelism": candidate.pipeline_parallelism,
            },
            "engine": {
                "name": candidate.serving_engine.value,
                "version": candidate.engine_version,
                "precision": candidate.precision.value,
                "max_batch_size": candidate.max_batch_size,
                "max_context_length": candidate.max_context_length,
            },
            "resources": {
                "expected_vram_gb": candidate.expected_vram_gb,
                "replicas": candidate.replica_count,
            },
            "rollback": {
                "strategy": "immediate",
            },
            "review_required": True,
        },
    }
    if recommendation:
        manifest["metadata"]["recommendation"] = recommendation.recommendation_id
        manifest["spec"]["economics"] = {
            "estimated_monthly_cost": str(recommendation.economic_assumptions.total_cost_monthly) if recommendation.economic_assumptions else "unknown",
            "estimated_monthly_revenue": str(recommendation.economic_assumptions.estimated_revenue_monthly) if recommendation.economic_assumptions else "unknown",
        }
    return manifest

def manifest_to_yaml(manifest: dict) -> str:
    return yaml.dump(manifest, default_flow_style=False)

def manifest_to_json(manifest: dict) -> str:
    return json.dumps(manifest, indent=2)
