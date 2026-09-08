from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    """Immutable FBS-first runtime contract."""

    model_config = ConfigDict(frozen=True)

    pipeline_version: str = Field(default="v0.3.0-fbs-ai", min_length=1)
    broker: Literal["FBS"] = "FBS"
    operational_timezone: str = "UTC"
    mt5_poll_seconds: float = Field(default=1.0, gt=0)
    historical_data_root: str = "data/fbs/historical"

    ai_enabled: bool = True
    ai_provider: Literal["ollama"] = "ollama"
    ai_model: str = "qwen3:4b"
    ai_base_url: str = "http://127.0.0.1:11434"
    ai_timeout_seconds: float = Field(default=20.0, gt=0)
    ai_fail_closed: bool = True

    def model_post_init(self, __context: object) -> None:
        if self.operational_timezone != "UTC":
            raise ValueError("Internal operational timezone must remain UTC")
        if not self.ai_model.strip():
            raise ValueError("ai_model must not be empty")
        if not self.ai_base_url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("Local AI endpoint must remain on localhost")
