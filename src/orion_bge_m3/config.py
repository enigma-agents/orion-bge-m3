from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_port: int = 9301
    log_level: str = "INFO"

    # Local path to BGE-M3 weights baked into the image at build time
    # via Dockerfile COPY. No network calls at startup.
    model_id: str = "/opt/orion/bge-m3"
    use_fp16: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
