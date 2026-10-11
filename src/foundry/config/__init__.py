from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "tesseract-foundry"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    database_url: str = "sqlite:///./foundry.db"
    api_key_enabled: bool = False
    api_key: Optional[str] = None
    max_job_concurrency: int = 4
    default_job_timeout_seconds: int = 3600
    profiling_cache_dir: str = "./profiling_cache"
    model_registry_dir: str = "./model_registry"
    dynamo_api_url: Optional[str] = None
    dynamo_namespace: str = "default"
    aiperf_binary_path: Optional[str] = None
    kubernetes_config_path: Optional[str] = None
    allowed_container_images: list[str] = []

    class Config:
        env_prefix = "FOUNDRY_"
        env_file = ".env"

settings = Settings()
