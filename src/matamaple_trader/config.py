from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    """Small immutable runtime contract used by early platform phases."""

    model_config = ConfigDict(frozen=True)

    pipeline_version: str = Field(default="v0.1.0", min_length=1)
    operational_timezone: str = "UTC"
    mt5_poll_seconds: float = Field(default=1.0, gt=0)

    def model_post_init(self, __context: object) -> None:
        if self.operational_timezone != "UTC":
            raise ValueError("Internal operational timezone must remain UTC")
