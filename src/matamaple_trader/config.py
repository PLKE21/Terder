from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    """Immutable FBS-first runtime contract."""

    model_config = ConfigDict(frozen=True)

    pipeline_version: str = Field(default="v0.2.0-fbs", min_length=1)
    broker: Literal["FBS"] = "FBS"
    operational_timezone: str = "UTC"
    mt5_poll_seconds: float = Field(default=1.0, gt=0)
    historical_data_root: str = "data/fbs/historical"

    def model_post_init(self, __context: object) -> None:
        if self.operational_timezone != "UTC":
            raise ValueError("Internal operational timezone must remain UTC")
