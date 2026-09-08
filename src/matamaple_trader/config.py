from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    """Immutable FBS-first runtime contract."""

    model_config = ConfigDict(frozen=True)

    pipeline_version: str = Field(default="v0.4.0-fbs-demo", min_length=1)
    broker: Literal["FBS"] = "FBS"
    operational_timezone: str = "UTC"
    mt5_poll_seconds: float = Field(default=1.0, gt=0)
    historical_data_root: str = "data/fbs/historical"

    ai_enabled: bool = True
    ai_provider: Literal["ollama"] = "ollama"
    ai_model: str = "qwen3:4b"
    ai_base_url: str = "http://127.0.0.1:11434"
    ai_timeout_seconds: float = Field(default=20.0, gt=0)
    ai_max_latency_seconds: float = Field(default=20.0, gt=0)
    ai_fail_closed: bool = True
    ai_max_consecutive_failures: int = Field(default=3, ge=1, le=20)
    ai_circuit_cooldown_seconds: float = Field(default=60.0, gt=0)

    execution_mode: Literal["DRY_RUN", "DEMO"] = "DRY_RUN"
    live_trading_enabled: Literal[False] = False

    def model_post_init(self, __context: object) -> None:
        if self.operational_timezone != "UTC":
            raise ValueError("Internal operational timezone must remain UTC")
        if not self.ai_model.strip():
            raise ValueError("ai_model must not be empty")

        parsed = urlparse(self.ai_base_url)
        if parsed.scheme != "http":
            raise ValueError("Local AI endpoint must use http")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Local AI endpoint must remain on loopback/localhost")
        if parsed.username or parsed.password:
            raise ValueError("Local AI endpoint must not embed credentials")
        if self.ai_max_latency_seconds > self.ai_timeout_seconds:
            raise ValueError("ai_max_latency_seconds cannot exceed ai_timeout_seconds")
