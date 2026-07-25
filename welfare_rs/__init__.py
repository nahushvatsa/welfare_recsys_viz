"""welfare_rs — welfare-oriented activity-travel agent-based model.

A self-contained simulation engine (no visualization, no web framework) for
modelling how recommender systems shape activity-travel behaviour and traveller
welfare on a real OpenStreetMap street network.

Quick start
-----------
    from welfare_rs import Simulation, build_metro_network

    net = build_metro_network("miami")     # two-layer metro (cached after first build)
    sim = Simulation(num_agents=80, seed=42, road_network=net,
                     disabled_modes=("transit",))
    day_summaries = sim.run_days(3)
    print(sim.summarize())

Legacy single-area presets remain available via ``build_road_network``.

The :mod:`welfare_rs.params` module is the central registry of tunable
assumptions; patch it before constructing a :class:`Simulation`.
"""

from __future__ import annotations

from . import params
from .agent import Agent
from .datasource import DataSource, LocalDataSource, PostgresDataSource, get_datasource
from .datastructures import Activity, Trip
from .geo import RoadNetwork, build_road_network, haversine_km
from .metro import build_metro_network, build_network, is_metro
from .recommender_systems import (
    GoogleMapsReplica,
    LeisureRSOrchestrator,
    Place,
    PlaceDynamics,
    PopularityRecommender,
    Recommendation,
    RecommenderSystem,
    UserContext,
    build_recommender_stack,
)
from .simulation import Simulation
from .welfare_layer import (
    ContinuousFeedback,
    TravelCostEstimator,
    WelfareAwareOrchestrator,
    compute_continuous_feedback,
    compute_expected_regret,
    compute_pup,
)

__version__ = "0.1.0"

__all__ = [
    "params",
    "Simulation",
    "Agent",
    "Activity",
    "Trip",
    "RoadNetwork",
    "build_road_network",
    "build_metro_network",
    "build_network",
    "is_metro",
    "haversine_km",
    # data access
    "DataSource",
    "LocalDataSource",
    "PostgresDataSource",
    "get_datasource",
    # recommenders
    "RecommenderSystem",
    "GoogleMapsReplica",
    "PopularityRecommender",
    "LeisureRSOrchestrator",
    "build_recommender_stack",
    "Place",
    "PlaceDynamics",
    "UserContext",
    "Recommendation",
    # welfare layer
    "WelfareAwareOrchestrator",
    "TravelCostEstimator",
    "ContinuousFeedback",
    "compute_continuous_feedback",
    "compute_pup",
    "compute_expected_regret",
]
