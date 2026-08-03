"""Request/response schemas for the API."""

from __future__ import annotations

import random
from typing import List, Optional

from pydantic import BaseModel, Field

from service import DEFAULT_CITY, MAX_SEEDS, TREATMENTS, RunConfig


class RunRequest(BaseModel):
    """Study config posted by the frontend.

    ``seed`` is the base seed; the study sweeps ``seed .. seed + num_seeds - 1``.
    It is **optional**: the UI does not send one, and a fresh base is drawn per
    submission so repeating a configuration gives a genuinely new sample.
    """

    city: str = DEFAULT_CITY
    # Ceiling is a guard against a typo'd request, not a modelling limit. Note
    # that map geometry retained per run scales as agents x days x seeds x
    # conditions (see viz.build_timelines), so large populations are bounded by
    # RAM long before this cap — run big populations with few seeds/days/
    # conditions, or fewer agents.
    num_agents: int = Field(80, ge=1, le=100000)
    num_days: int = Field(3, ge=1, le=60)
    # None (the UI's default) means "draw a fresh base seed for this run".
    # A fixed seed did more than make runs repeatable: run_id() hashes the
    # config, so an identical request resolved to an identical id and the
    # manager handed back the CACHED study rather than simulating anything.
    # Re-running the same settings could not produce a different answer.
    seed: Optional[int] = None
    num_seeds: int = Field(3, ge=1, le=MAX_SEEDS)
    # Recommenders to run. The No-RS control is always added on top by
    # RunConfig.all_conditions(), so an empty list is a valid request meaning
    # "baseline only".
    conditions: List[str] = Field(default_factory=lambda: ["Standard RS"])
    # Deprecated single-condition alias, kept so older clients and saved API
    # calls keep working. See to_config() for the precedence rule.
    treatment: Optional[str] = None
    multimodal: bool = False
    use_real_pois: bool = True
    pup_alpha: float = Field(0.6, ge=0.0, le=1.0)
    rm_epsilon: float = Field(0.3, ge=0.0, le=1.5)

    def to_config(self) -> RunConfig:
        # `treatment` applies only when the caller did not send `conditions` at
        # all — checking model_fields_set rather than truthiness, so a legacy
        # POST still means exactly what it used to and an explicit empty
        # `conditions: []` (baseline only) is never overridden by the alias.
        if "conditions" not in self.model_fields_set and self.treatment is not None:
            requested = [self.treatment]
        else:
            requested = list(self.conditions)
        # Unknown names are dropped rather than rejected; all_conditions() then
        # dedupes, prepends No RS, and puts them in canonical order.
        conditions = tuple(c for c in requested if c in TREATMENTS)

        # Resolved to a concrete int here, once, so the whole study — every
        # worker, the run id, the cached record — agrees on it, and so the
        # seeds actually used come back in the run payload. Reproducibility is
        # not lost, just moved: POST the same seed explicitly to replay a study.
        # Headroom of MAX_SEEDS keeps the sweep from overflowing the range.
        seed = self.seed
        if seed is None:
            seed = random.randrange(1, 2**31 - MAX_SEEDS)
        return RunConfig(
            city=self.city,
            num_agents=self.num_agents,
            num_days=self.num_days,
            seed=seed,
            num_seeds=self.num_seeds,
            conditions=conditions,
            multimodal=self.multimodal,
            use_real_pois=self.use_real_pois,
            pup_alpha=self.pup_alpha,
            rm_epsilon=self.rm_epsilon,
        )
