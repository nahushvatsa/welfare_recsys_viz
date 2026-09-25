"""Request/response schemas for the API."""

from __future__ import annotations

import random
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from service import (DEFAULT_CITY, MAX_RECOMMENDERS, MAX_SEEDS, TREATMENTS,
                     RecommenderSpec, RunConfig, WELFARE_GATES)


class RecommenderSpecRequest(BaseModel):
    """One recommender the user built in the sidebar.

    Every field is a knob on the *same* scorer — the platforms differ by which
    weights they zero out, not by having different machinery. Weights need not
    sum to anything: they are renormalized engine-side, so the sliders can move
    independently and the UI can show the resulting percentages.
    """

    label: str = Field("Custom RS", min_length=1, max_length=40)
    w_rating: float = Field(0.20, ge=0.0, le=1.0)
    w_reviews: float = Field(0.15, ge=0.0, le=1.0)
    w_relevance: float = Field(0.10, ge=0.0, le=1.0)
    w_proximity: float = Field(0.25, ge=0.0, le=1.0)
    w_popularity: float = Field(0.20, ge=0.0, le=1.0)
    w_personalization: float = Field(0.10, ge=0.0, le=1.0)
    distance_scale_km: float = Field(5.0, gt=0.0, le=50.0)
    popularity_gamma: float = Field(1.0, ge=0.0, le=3.0)
    learning_rate: float = Field(0.12, ge=0.0, le=1.0)
    #: "" | "pup" | "rm" | "pup_rm" — the welfare gate applied on top.
    welfare_gate: str = ""
    pup_alpha: float = Field(0.6, ge=0.0, le=1.0)
    rm_epsilon: float = Field(0.3, ge=0.0, le=1.5)
    random_ranking: bool = False

    @field_validator("welfare_gate")
    @classmethod
    def _known_gate(cls, value: str) -> str:
        gate = (value or "").strip().lower()
        return gate if gate in WELFARE_GATES else ""

    def to_spec(self) -> RecommenderSpec:
        return RecommenderSpec(
            label=self.label.strip() or "Custom RS",
            w_rating=self.w_rating, w_reviews=self.w_reviews,
            w_relevance=self.w_relevance, w_proximity=self.w_proximity,
            w_popularity=self.w_popularity,
            w_personalization=self.w_personalization,
            distance_scale_km=self.distance_scale_km,
            popularity_gamma=self.popularity_gamma,
            learning_rate=self.learning_rate,
            welfare_gate=self.welfare_gate,
            pup_alpha=self.pup_alpha, rm_epsilon=self.rm_epsilon,
            random_ranking=self.random_ranking,
        )


class RunRequest(BaseModel):
    """Study config posted by the frontend.

    ``seed`` is the base seed; the study sweeps ``seed .. seed + num_seeds - 1``.
    It is **optional**: the UI does not send one, and a fresh base is drawn per
    submission so repeating a configuration gives a genuinely new sample.
    """

    city: str = DEFAULT_CITY
    # Map geometry retained per run scales as agents x days x seeds x
    # conditions (see viz.build_timelines), so large populations are bounded by
    # RAM long before this cap — run big populations with few seeds/days/
    # conditions, or fewer agents.
    #
    # A guard against a typo'd request, not a modelling limit: cost goes as
    # agents x days x seeds x conditions, and the operator is the one who knows
    # what this box can afford. Deliberately NOT a wall-clock budget check —
    # any such estimate has to model the metro network build (minutes to tens of
    # minutes on an unwarmed metro), which is exactly the term it cannot see.
    num_agents: int = Field(80, ge=1, le=10000)
    # Measured, not guessed. Agents take ~0.15 leisure outings per day and
    # accept roughly 60% of recommendations, so a study yields about
    # 0.09 feedback events per agent per day — and personalization learns from
    # nothing else. At 10 days that is ~0.9 events per agent, most of them zero
    # or one, and the taste-discovery curve is flat whatever the recommender is
    # configured to do. (The learning itself is not weak: in isolation a single
    # like is enough to fill the top of the slate with matching venues.) 30 days
    # puts ~2.7 events on the average agent, which is where the curve starts to
    # move. Days are the cheap axis — the per-task routing precompute dominates.
    num_days: int = Field(30, ge=1, le=60)
    # None (the UI's default) means "draw a fresh base seed for this run".
    # A fixed seed did more than make runs repeatable: run_id() hashes the
    # config, so an identical request resolved to an identical id and the
    # manager handed back the CACHED study rather than simulating anything.
    # Re-running the same settings could not produce a different answer.
    seed: Optional[int] = None
    num_seeds: int = Field(3, ge=1, le=MAX_SEEDS)
    # Recommenders the user built. The No-RS control is always added on top by
    # RunConfig.all_conditions(), so an empty list is a valid request meaning
    # "baseline only".
    recommenders: List[RecommenderSpecRequest] = Field(default_factory=list)
    # Deprecated: the fixed five-treatment vocabulary this replaced. Kept so
    # saved API calls keep working — each surviving name resolves to the
    # equivalent spec (see service.spec_from_legacy_name). Ignored whenever
    # `recommenders` is sent.
    conditions: List[str] = Field(default_factory=list)
    treatment: Optional[str] = None
    use_real_pois: bool = True
    pup_alpha: float = Field(0.6, ge=0.0, le=1.0)
    rm_epsilon: float = Field(0.3, ge=0.0, le=1.5)

    @field_validator("recommenders")
    @classmethod
    def _cap_and_dedupe(
        cls, value: List[RecommenderSpecRequest]
    ) -> List[RecommenderSpecRequest]:
        """Keep the first spec per label, up to MAX_RECOMMENDERS.

        Labels are the identity of a study arm everywhere downstream — the
        results tables, the map's condition picker, the run id — so two arms
        sharing one would collapse into each other silently. Later duplicates
        are suffixed rather than dropped, since the user did ask for two.
        """
        out: List[RecommenderSpecRequest] = []
        seen: dict = {}
        for spec in value[:MAX_RECOMMENDERS]:
            label = (spec.label or "Custom RS").strip() or "Custom RS"
            if label in seen:
                seen[label] += 1
                label = f"{label} ({seen[label]})"
            else:
                seen[label] = 1
            out.append(spec.model_copy(update={"label": label}))
        return out

    def _legacy_specs(self) -> List[RecommenderSpec]:
        """Specs implied by the deprecated `conditions` / `treatment` fields."""
        from service import spec_from_legacy_name

        if "conditions" not in self.model_fields_set and self.treatment is not None:
            requested = [self.treatment]
        else:
            requested = list(self.conditions)
        specs = []
        for name in requested:
            if name not in TREATMENTS:
                continue
            spec = spec_from_legacy_name(name, self.pup_alpha, self.rm_epsilon)
            if spec is not None:
                specs.append(spec)
        return specs

    def to_specs(self) -> List[RecommenderSpec]:
        """The recommenders this study should run, No-RS control excluded."""
        if self.recommenders:
            return [r.to_spec() for r in self.recommenders]
        return self._legacy_specs()

    def to_config(self) -> RunConfig:
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
            recommenders=tuple(self.to_specs()),
            use_real_pois=self.use_real_pois,
        )
