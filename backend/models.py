"""Request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from service import DEFAULT_CITY, MAX_SEEDS, TREATMENTS, RunConfig


class RunRequest(BaseModel):
    """Study config posted by the frontend.

    ``seed`` is the base seed; the study sweeps ``seed .. seed + num_seeds - 1``.
    """

    city: str = DEFAULT_CITY
    num_agents: int = Field(80, ge=1, le=5000)
    num_days: int = Field(3, ge=1, le=60)
    seed: int = 42
    num_seeds: int = Field(3, ge=1, le=MAX_SEEDS)
    treatment: str = "Standard RS"
    multimodal: bool = False
    use_real_pois: bool = True
    pup_alpha: float = Field(0.6, ge=0.0, le=1.0)
    rm_epsilon: float = Field(0.3, ge=0.0, le=1.5)

    def to_config(self) -> RunConfig:
        treatment = self.treatment if self.treatment in TREATMENTS else "Standard RS"
        return RunConfig(
            city=self.city,
            num_agents=self.num_agents,
            num_days=self.num_days,
            seed=self.seed,
            num_seeds=self.num_seeds,
            treatment=treatment,
            multimodal=self.multimodal,
            use_real_pois=self.use_real_pois,
            pup_alpha=self.pup_alpha,
            rm_epsilon=self.rm_epsilon,
        )
