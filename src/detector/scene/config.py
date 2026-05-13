from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Zone(BaseModel):
    name: str
    policy: Literal["forbidden"]
    points: list[tuple[float, float]]

    @field_validator("points")
    @classmethod
    def _check_points(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("Zone polygon must have at least 3 points")
        return v


class CrowdCfg(BaseModel):
    min_people: int = Field(ge=2)
    radius_px: float = Field(gt=0)


class SceneConfig(BaseModel):
    source: str
    frame_size: tuple[int, int]
    zones: list[Zone] = []
    crowd: CrowdCfg | None = None

    @model_validator(mode="after")
    def _check_any(self) -> SceneConfig:
        if not self.zones and self.crowd is None:
            raise ValueError("SceneConfig must define at least one zone or crowd config")
        return self


def load_scene(path: str | None) -> SceneConfig | None:
    """Load a scene JSON. Returns None when path is empty/None.

    Raises FileNotFoundError when path is set but the file is missing — a wrong
    path is always a bug, not a soft-disable.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scene config not found: {path}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return SceneConfig.model_validate(data)
