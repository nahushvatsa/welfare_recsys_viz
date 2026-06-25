"""Recommender-system module for leisure activity recommendations.

This file implements:
- A Google Maps replica recommender based on Prominence, Relevance, Proximity.
- Popularity-based recommenders for OpenTable, Spotify/Ticketmaster, and ClassPass.

The module is intentionally independent from the ABM notebook so it can be plugged
into agent decision logic later.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


Coordinate = Tuple[float, float]


LEISURE_SUBTYPE_TO_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "food_takeout": ("food_takeout", "restaurant", "food"),
    "food_dine_in": ("restaurant", "food"),
    "live_music": ("live_music", "concert_venue", "music_event"),
    "workout_or_run": ("fitness_studio", "gym", "running_route"),
    "cafe_friend": ("cafe",),
    "museum": ("museum",),
    "park": ("park",),
}


LEISURE_SUBTYPE_TO_DEFAULT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "food_takeout": ("takeout", "quick", "food"),
    "food_dine_in": ("restaurant", "dinner", "food"),
    "live_music": ("live", "concert", "music"),
    "workout_or_run": ("workout", "fitness", "run"),
    "cafe_friend": ("cafe", "coffee", "friends"),
    "museum": ("museum", "art", "culture"),
    "park": ("park", "green", "outdoor"),
}


@dataclass(frozen=True)
class Place:
    """A candidate place or event that can be recommended."""

    place_id: str
    name: str
    category: str
    location: Coordinate
    keywords: Tuple[str, ...] = field(default_factory=tuple)
    rating: float = 0.0
    review_count: int = 0
    popularity: float = 0.0


@dataclass(frozen=True)
class UserContext:
    """User context passed into recommenders."""

    user_id: int
    location: Coordinate
    query_keywords: Tuple[str, ...] = field(default_factory=tuple)
    interest_keywords: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Recommendation:
    """Scored recommendation with score breakdown for transparency."""

    source: str
    place: Place
    score: float
    components: Dict[str, float]


def euclidean_distance_km(a: Coordinate, b: Coordinate) -> float:
    """Approximate Euclidean distance in grid units interpreted as km."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _normalize(values: Sequence[float]) -> List[float]:
    """Min-max normalize a sequence; return 1.0 for all if constant."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _token_overlap_score(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    """Return overlap score in [0, 1] using Jaccard similarity."""
    set_a = {t.lower().strip() for t in tokens_a if t}
    set_b = {t.lower().strip() for t in tokens_b if t}
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / max(1, union)


class RecommenderSystem(ABC):
    """Base interface for recommenders."""

    def __init__(self, name: str, catalog: Sequence[Place]):
        self.name = name
        self.catalog = list(catalog)
        self.place_by_id = {p.place_id: p for p in self.catalog}
        self.user_place_affinity: Dict[int, Dict[str, float]] = {}
        self.user_keyword_affinity: Dict[int, Dict[str, float]] = {}

    @abstractmethod
    def recommend(
        self,
        user: UserContext,
        leisure_subtype: str,
        top_k: int = 5,
    ) -> List[Recommendation]:
        """Return ranked recommendations for a leisure subtype."""

    def _filter_by_subtype(self, leisure_subtype: str) -> List[Place]:
        allowed = LEISURE_SUBTYPE_TO_CATEGORIES.get(leisure_subtype, ())
        if not allowed:
            return []
        return [p for p in self.catalog if p.category in allowed]

    def _personalization_score(self, user_id: int, place: Place) -> float:
        """Return personalized score in [0, 1] from learned user affinities."""
        place_affinity = self.user_place_affinity.get(user_id, {}).get(place.place_id, 0.0)
        kw_aff = self.user_keyword_affinity.get(user_id, {})
        if place.keywords:
            kw_vals = [kw_aff.get(k.lower(), 0.0) for k in place.keywords]
            kw_affinity = sum(kw_vals) / len(kw_vals)
        else:
            kw_affinity = 0.0
        # place_affinity and kw_affinity are in [-1, 1]. Map to [0, 1].
        combined = 0.6 * place_affinity + 0.4 * kw_affinity
        return max(0.0, min(1.0, 0.5 * (combined + 1.0)))

    def record_feedback(
        self,
        user_id: int,
        place_id: str,
        liked: bool,
        feedback_strength: float = 1.0,
        learning_rate: float = 0.12,
    ) -> None:
        """Update user personalization state from feedback."""
        place = self.place_by_id.get(place_id)
        if place is None:
            return
        delta = learning_rate * max(0.2, min(2.0, feedback_strength))
        if not liked:
            delta *= -1.0

        up = self.user_place_affinity.setdefault(user_id, {})
        old_place = up.get(place_id, 0.0)
        up[place_id] = max(-1.0, min(1.0, old_place + delta))

        uk = self.user_keyword_affinity.setdefault(user_id, {})
        for keyword in place.keywords:
            k = keyword.lower()
            old_kw = uk.get(k, 0.0)
            uk[k] = max(-1.0, min(1.0, old_kw + 0.7 * delta))


class GoogleMapsReplica(RecommenderSystem):
    """Google Maps-like ranking with Prominence, Relevance, and Proximity.

    Prominence:
      Uses rating and review count.
    Relevance:
      Keyword overlap between place keywords and user query/interests.
    Proximity:
      Exponential decay with distance from user location.
    """

    def __init__(
        self,
        catalog: Sequence[Place],
        prominence_weight: float = 0.45,
        relevance_weight: float = 0.35,
        proximity_weight: float = 0.20,
        personalization_weight: float = 0.10,
        distance_scale_km: float = 5.0,
        coord_scale_km: float = 1.0,
        coord_distance_km=euclidean_distance_km,
    ):
        super().__init__(name="google_maps", catalog=catalog)
        weight_sum = prominence_weight + relevance_weight + proximity_weight
        self.prominence_weight = prominence_weight / weight_sum
        self.relevance_weight = relevance_weight / weight_sum
        self.proximity_weight = proximity_weight / weight_sum
        self.personalization_weight = max(0.0, min(0.35, personalization_weight))
        self.distance_scale_km = max(0.1, distance_scale_km)
        self.coord_scale_km = max(1e-4, coord_scale_km)
        # Distance between two location tuples. Defaults to Euclidean (grid mode);
        # OSM mode injects haversine so (lat, lon) proximity is measured in km.
        self.coord_distance_km = coord_distance_km

    def _prominence_scores(self, places: Sequence[Place]) -> Dict[str, float]:
        if not places:
            return {}
        rating_signal = [max(0.0, min(5.0, p.rating)) / 5.0 for p in places]
        review_signal_raw = [math.log1p(max(0, p.review_count)) for p in places]
        review_signal = _normalize(review_signal_raw)
        prominence = [
            0.55 * rating_signal[i] + 0.45 * review_signal[i]
            for i in range(len(places))
        ]
        return {places[i].place_id: prominence[i] for i in range(len(places))}

    def _relevance_score(self, place: Place, user: UserContext, leisure_subtype: str) -> float:
        query_tokens = list(user.query_keywords)
        if not query_tokens:
            query_tokens = list(LEISURE_SUBTYPE_TO_DEFAULT_KEYWORDS.get(leisure_subtype, ()))
        interest_tokens = list(user.interest_keywords)
        qr_score = _token_overlap_score(place.keywords, query_tokens)
        qi_score = _token_overlap_score(place.keywords, interest_tokens)
        return 0.75 * qr_score + 0.25 * qi_score

    def _proximity_score(self, place: Place, user: UserContext) -> float:
        dist_km = self.coord_distance_km(user.location, place.location) * self.coord_scale_km
        return math.exp(-dist_km / self.distance_scale_km)

    def recommend(self, user: UserContext, leisure_subtype: str, top_k: int = 5) -> List[Recommendation]:
        candidates = self._filter_by_subtype(leisure_subtype)
        if not candidates:
            return []
        prominence_by_id = self._prominence_scores(candidates)
        scored: List[Recommendation] = []
        for place in candidates:
            prominence = prominence_by_id.get(place.place_id, 0.0)
            relevance = self._relevance_score(place, user, leisure_subtype)
            proximity = self._proximity_score(place, user)
            base_score = (
                self.prominence_weight * prominence
                + self.relevance_weight * relevance
                + self.proximity_weight * proximity
            )
            personalized = self._personalization_score(user.user_id, place)
            score = (1.0 - self.personalization_weight) * base_score + self.personalization_weight * personalized
            scored.append(
                Recommendation(
                    source=self.name,
                    place=place,
                    score=score,
                    components={
                        "prominence": prominence,
                        "relevance": relevance,
                        "proximity": proximity,
                        "personalization": personalized,
                    },
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(1, top_k)]


class PopularityRecommender(RecommenderSystem):
    """Popularity-only ranking (for OpenTable, Spotify/Ticketmaster, ClassPass)."""

    def __init__(
        self,
        name: str,
        catalog: Sequence[Place],
        rating_weight: float = 0.25,
        review_weight: float = 0.35,
        popularity_weight: float = 0.40,
        personalization_weight: float = 0.12,
    ):
        super().__init__(name=name, catalog=catalog)
        weight_sum = rating_weight + review_weight + popularity_weight
        self.rating_weight = rating_weight / weight_sum
        self.review_weight = review_weight / weight_sum
        self.popularity_weight = popularity_weight / weight_sum
        self.personalization_weight = max(0.0, min(0.35, personalization_weight))

    def _popularity_score(self, place: Place) -> float:
        rating_term = max(0.0, min(5.0, place.rating)) / 5.0
        review_term = math.log1p(max(0, place.review_count))
        pop_term = math.log1p(max(0.0, place.popularity))
        # Keep this popularity-focused: no proximity/relevance terms here.
        return (
            self.rating_weight * rating_term
            + self.review_weight * review_term
            + self.popularity_weight * pop_term
        )

    def recommend(self, user: UserContext, leisure_subtype: str, top_k: int = 5) -> List[Recommendation]:
        candidates = self._filter_by_subtype(leisure_subtype)
        if not candidates:
            return []
        raw_scores = [self._popularity_score(p) for p in candidates]
        norm_scores = _normalize(raw_scores)
        scored = []
        for i, place in enumerate(candidates):
            personalized = self._personalization_score(user.user_id, place)
            score = (1.0 - self.personalization_weight) * norm_scores[i] + self.personalization_weight * personalized
            scored.append(
                Recommendation(
                    source=self.name,
                    place=place,
                    score=score,
                    components={"popularity": norm_scores[i], "personalization": personalized},
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(1, top_k)]


class LeisureRSOrchestrator:
    """Routes each leisure subtype to the requested recommender platforms."""

    def __init__(
        self,
        google_maps_rs: GoogleMapsReplica,
        opentable_rs: PopularityRecommender,
        spotify_ticketmaster_rs: PopularityRecommender,
        classpass_rs: PopularityRecommender,
    ):
        self.google_maps_rs = google_maps_rs
        self.opentable_rs = opentable_rs
        self.spotify_ticketmaster_rs = spotify_ticketmaster_rs
        self.classpass_rs = classpass_rs

    def recommend(
        self,
        user: UserContext,
        leisure_subtype: str,
        top_k_per_system: int = 5,
    ) -> Dict[str, List[Recommendation]]:
        """Return recommendations by platform for a leisure subtype."""
        out: Dict[str, List[Recommendation]] = {}
        if leisure_subtype in {"food_takeout", "food_dine_in"}:
            out[self.google_maps_rs.name] = self.google_maps_rs.recommend(user, leisure_subtype, top_k_per_system)
            out[self.opentable_rs.name] = self.opentable_rs.recommend(user, leisure_subtype, top_k_per_system)
        elif leisure_subtype == "live_music":
            out[self.spotify_ticketmaster_rs.name] = self.spotify_ticketmaster_rs.recommend(
                user, leisure_subtype, top_k_per_system
            )
        elif leisure_subtype == "workout_or_run":
            out[self.classpass_rs.name] = self.classpass_rs.recommend(user, leisure_subtype, top_k_per_system)
        elif leisure_subtype in {"museum", "cafe_friend", "park"}:
            out[self.google_maps_rs.name] = self.google_maps_rs.recommend(user, leisure_subtype, top_k_per_system)
        return out

    def record_feedback(
        self,
        user_id: int,
        source: str,
        place_id: str,
        liked: bool,
        feedback_strength: float = 1.0,
    ) -> None:
        """Update the corresponding recommender's user model from feedback."""
        recommender = None
        if source == self.google_maps_rs.name:
            recommender = self.google_maps_rs
        elif source == self.opentable_rs.name:
            recommender = self.opentable_rs
        elif source == self.spotify_ticketmaster_rs.name:
            recommender = self.spotify_ticketmaster_rs
        elif source == self.classpass_rs.name:
            recommender = self.classpass_rs
        if recommender is None:
            return
        recommender.record_feedback(
            user_id=user_id,
            place_id=place_id,
            liked=liked,
            feedback_strength=feedback_strength,
        )


def build_recommender_stack(
    catalog: Sequence[Place],
    google_maps_config: Dict[str, float] | None = None,
    popularity_config: Dict[str, float] | None = None,
) -> LeisureRSOrchestrator:
    """Convenience builder for the RS stack using a shared place catalog."""
    google_maps_config = google_maps_config or {}
    popularity_config = popularity_config or {}
    return LeisureRSOrchestrator(
        google_maps_rs=GoogleMapsReplica(catalog=catalog, **google_maps_config),
        opentable_rs=PopularityRecommender(name="opentable", catalog=catalog, **popularity_config),
        spotify_ticketmaster_rs=PopularityRecommender(
            name="spotify_ticketmaster", catalog=catalog, **popularity_config
        ),
        classpass_rs=PopularityRecommender(name="classpass", catalog=catalog, **popularity_config),
    )
