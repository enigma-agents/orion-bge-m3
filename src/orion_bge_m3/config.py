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

    # Compute backend. "auto" picks mps (Apple Silicon Metal) > cuda >
    # cpu at startup. Override with BGE_M3_DEVICE={cpu,cuda,mps,auto}.
    # NOTE: mps only works when the service runs as a NATIVE macOS
    # process — Docker containers on macOS are Linux guests with no
    # Metal device, so inside a container this resolves to cpu.
    device: str = "auto"

    # fp16 halves memory (~2.3GB fp32 -> ~1.4GB) with negligible quality
    # loss. Left False so the default (auto -> may pick cpu, where fp16 is
    # slow/unsupported) is safe; the encoder force-enables it on mps.
    use_fp16: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
