from ..models import DeploymentCandidate, ServiceRecommendation
from typing import Optional
import yaml
import json


def generate_dynamo_manifest(candidate: DeploymentCandidate) -> dict:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": f"foundry-{candidate.model_id.replace('/', '-')[:40]}",
            "labels": {
                "app.kubernetes.io/name": "tesseract-foundry",
                "app.kubernetes.io/managed-by": "foundry",
                "foundry.ai/model": candidate.model_id.replace("/", "-"),
                "foundry.ai/candidate": candidate.candidate_id,
                "foundry.ai/review-required": "true",
            },
            "annotations": {
                "foundry.ai/engine": candidate.serving_engine.value,
                "foundry.ai/engine-version": candidate.engine_version,
                "foundry.ai/precision": candidate.precision.value,
                "foundry.ai/tensor-parallelism": str(candidate.tensor_parallelism),
                "foundry.ai/gpu-count": str(candidate.gpu_count),
            },
        },
        "spec": {
            "model": {
                "id": candidate.model_id,
                "revision": getattr(candidate, "model_revision", "latest"),
                "source": getattr(candidate, "model_source", "huggingface"),
            },
            "hardware": {
                "gpuCount": candidate.gpu_count,
                "tensorParallelism": candidate.tensor_parallelism,
                "pipelineParallelism": candidate.pipeline_parallelism,
            },
            "engine": {
                "name": candidate.serving_engine.value,
                "version": candidate.engine_version,
                "image": f"nvcr.io/nvidia/{candidate.serving_engine.value}:{candidate.engine_version}",
                "imagePullPolicy": "IfNotPresent",
                "precision": candidate.precision.value,
                "maxBatchSize": candidate.max_batch_size,
                "maxContextLength": candidate.max_context_length,
            },
            "resources": {
                "limits": {
                    "nvidia.com/gpu": candidate.gpu_count,
                    "memory": f"{int(candidate.expected_vram_gb * 1.2)}Gi",
                },
                "requests": {
                    "nvidia.com/gpu": candidate.gpu_count,
                    "memory": f"{int(candidate.expected_vram_gb * 1.1)}Gi",
                },
            },
            "replicas": candidate.replica_count,
            "health": {
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "initialDelaySeconds": 30,
                    "periodSeconds": 10,
                },
                "livenessProbe": {
                    "httpGet": {"path": "/healthz", "port": 8080},
                    "initialDelaySeconds": 60,
                    "periodSeconds": 30,
                },
            },
            "rollback": {
                "strategy": "immediate",
                "revisionHistoryLimit": 3,
            },
            "autoDeploy": False,
            "reviewRequired": True,
        },
    }


def generate_kubernetes_manifest(candidate: DeploymentCandidate) -> dict:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"foundry-{candidate.model_id.replace('/', '-')[:40]}",
            "labels": {
                "app": "tesseract-foundry",
                "model": candidate.model_id.replace("/", "-"),
                "candidate": candidate.candidate_id,
            },
            "annotations": {
                "foundry.ai/managed": "true",
                "foundry.ai/review-required": "true",
            },
        },
        "spec": {
            "replicas": candidate.replica_count,
            "selector": {"matchLabels": {"app": "foundry-model"}},
            "template": {
                "metadata": {"labels": {"app": "foundry-model"}},
                "spec": {
                    "containers": [{
                        "name": "inference",
                        "image": f"nvcr.io/nvidia/{candidate.serving_engine.value}:{candidate.engine_version}",
                        "imagePullPolicy": "IfNotPresent",
                        "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                        "env": [
                            {"name": "MODEL_ID", "value": candidate.model_id},
                            {"name": "TENSOR_PARALLEL_SIZE", "value": str(candidate.tensor_parallelism)},
                            {"name": "PRECISION", "value": candidate.precision.value},
                            {"name": "MAX_MODEL_LEN", "value": str(candidate.max_context_length)},
                        ],
                        "resources": {
                            "limits": {"nvidia.com/gpu": candidate.gpu_count, "memory": f"{int(candidate.expected_vram_gb * 1.2)}Gi"},
                            "requests": {"nvidia.com/gpu": candidate.gpu_count, "memory": f"{int(candidate.expected_vram_gb * 1.1)}Gi"},
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10,
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/healthz", "port": 8000},
                            "initialDelaySeconds": 60,
                            "periodSeconds": 30,
                        },
                    }],
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [{
                                    "matchExpressions": [{
                                        "key": "nvidia.com/gpu.product",
                                        "operator": "In",
                                        "values": [candidate.hardware_id],
                                    }],
                                }],
                            },
                        },
                    },
                    "tolerations": [{
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }],
                },
            },
        },
    }


def generate_manifest(candidate: DeploymentCandidate, recommendation: Optional[ServiceRecommendation] = None, format: str = "dynamo") -> dict:
    if format == "kubernetes":
        manifest = generate_kubernetes_manifest(candidate)
    else:
        manifest = generate_dynamo_manifest(candidate)

    if recommendation and recommendation.economic_assumptions:
        econ = recommendation.economic_assumptions
        manifest.setdefault("metadata", {}).setdefault("annotations", {})
        manifest["metadata"]["annotations"]["foundry.ai/recommendation-id"] = recommendation.recommendation_id
        manifest["metadata"]["annotations"]["foundry.ai/estimated-monthly-cost"] = str(econ.total_cost_monthly)
        manifest["metadata"]["annotations"]["foundry.ai/estimated-monthly-revenue"] = str(econ.estimated_revenue_monthly)
        manifest["metadata"]["annotations"]["foundry.ai/margin-pct"] = str(econ.margin_pct)

    return manifest


def manifest_to_yaml(manifest: dict) -> str:
    return yaml.dump(manifest, default_flow_style=False)


def manifest_to_json(manifest: dict) -> str:
    return json.dumps(manifest, indent=2)


def dry_run_validate(manifest: dict) -> dict:
    errors = []
    warnings = []
    if manifest.get("spec", {}).get("autoDeploy") is not False:
        warnings.append("autoDeploy should be false for review workflow")
    resources = manifest.get("spec", {}).get("resources", {})
    if resources:
        gpu_limit = resources.get("limits", {}).get("nvidia.com/gpu", 0)
        if gpu_limit == 0:
            errors.append("No GPU resource limit specified")
    container = manifest.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [{}])[0]
    if container.get("readinessProbe") is None and container.get("livenessProbe") is None:
        warnings.append("No health probes configured")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
