from ..models import HardwareProfile, PrecisionType, HardwareVendor
from typing import Optional

NVIDIA_GPU_CATALOG: dict[str, dict] = {
    "h100": {"vram": 80, "vendor": "nvidia", "precision": ["fp16", "bf16", "fp8", "int8"]},
    "h100sxm": {"vram": 80, "vendor": "nvidia", "precision": ["fp16", "bf16", "fp8", "int8"]},
    "a100": {"vram": 80, "vendor": "nvidia", "precision": ["fp16", "bf16", "int8"]},
    "a10040gb": {"vram": 40, "vendor": "nvidia", "precision": ["fp16", "bf16", "int8"]},
    "a6000": {"vram": 48, "vendor": "nvidia", "precision": ["fp16", "int8"]},
    "l40s": {"vram": 48, "vendor": "nvidia", "precision": ["fp16", "bf16", "int8"]},
    "l4": {"vram": 24, "vendor": "nvidia", "precision": ["fp16", "int8"]},
    "t4": {"vram": 16, "vendor": "nvidia", "precision": ["fp16", "int8"]},
}

def lookup_gpu(gpu_model: str) -> Optional[dict]:
    key = gpu_model.lower().replace("-", "").replace(" ", "")
    for catalog_key, spec in NVIDIA_GPU_CATALOG.items():
        if key == catalog_key or key == catalog_key.replace("-", ""):
            return spec
    return None

def check_vram_fit(model_params_millions: Optional[int], precision_bits: int, context_length: int, vram_gb: float, overhead_pct: float = 0.15) -> tuple[bool, float]:
    if not model_params_millions:
        return True, 0
    params = model_params_millions * 1e6
    weight_bytes = params * (precision_bits / 8)
    kv_cache_per_token_bytes = 2 * 2 * (model_params_millions / 1e6) * context_length * 2  # ~2 key/value heads * 2 bytes * hidden_dim ~ 2 * params_per_layer
    kv_cache_bytes = min(kv_cache_per_token_bytes, params * 0.5)  # cap at 50% of weights
    total_bytes = weight_bytes + kv_cache_bytes
    total_gb = total_bytes / (1024**3) * (1 + overhead_pct)
    available_gb = vram_gb * 0.9
    return total_gb <= available_gb, round(total_gb, 2)

def check_backend_compatibility(hw: HardwareProfile, backend: str) -> tuple[bool, str]:
    if hw.vendor != HardwareVendor.nvidia:
        return False, f"Backend {backend} not supported on {hw.vendor}"
    return True, "compatible"

def check_precision_support(hw: HardwareProfile, precision: PrecisionType) -> tuple[bool, str]:
    if precision.value not in [p.value for p in hw.supported_precision]:
        spec = lookup_gpu(hw.gpu_model)
        if spec and precision.value in spec["precision"]:
            return True, "compatible"
        return False, f"Precision {precision} not supported on {hw.gpu_model}"
    return True, "compatible"
