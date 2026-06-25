"""City spatial environment for the travel-behaviour ABM."""

from __future__ import annotations

import copy
import random

import numpy as np

from . import params


class City:
    """Spatial environment and global context.

    The city is a grid. Each cell has a zone type. The context includes
    policy and environmental conditions, plus social norms and practices.
    """

    def __init__(self, size_x, size_y, block_km=None, seed=42, road_network=None):
        cp = params.CITY_PARAMS
        self.size_x = size_x
        self.size_y = size_y
        self.block_km = block_km if block_km is not None else cp["block_km"]
        self.rng = random.Random(seed)
        # When set, the city is backed by a real OSM street network (see geo.py)
        # and locations are (lat, lon) node coordinates rather than grid cells.
        self.road_network = road_network

        # Context factors (C) from MTTC — deep-copy so runtime patches don't
        # mutate the template in params.py.
        self.context = copy.deepcopy(cp["default_context"])
        self.context["community_mode_bias"] = dict(cp["community_mode_bias"])

        if road_network is None:
            self._build_zones()
            area = size_x * size_y
        else:
            # OSM mode: the grid/zone machinery is unused; sampling and distance
            # delegate to the road network. Capacity scales with network size.
            self.zones = None
            self.zone_index = {}
            area = road_network.num_nodes

        self.road_capacity = max(
            cp["road_capacity_floor"],
            int(area * cp["road_capacity_multiplier"]),
        )
        self.transit_capacity = max(
            cp["transit_capacity_floor"],
            int(area * cp["transit_capacity_multiplier"]),
        )

    def _build_zones(self):
        """Assign a zone type to each grid cell using a simple probability mix."""
        probs = params.CITY_PARAMS["zone_probabilities"]
        choices = list(probs.keys())
        weights = [probs[k] for k in choices]
        zones = []
        for _ in range(self.size_x * self.size_y):
            zones.append(self.rng.choices(choices, weights=weights, k=1)[0])

        self.zones = np.array(zones, dtype=object).reshape(self.size_x, self.size_y)
        self.zone_index = {k: [] for k in choices}
        for x in range(self.size_x):
            for y in range(self.size_y):
                self.zone_index[self.zones[x, y]].append((x, y))

    def distance_km(self, a, b):
        """Return travel distance in kilometers between two locations.

        Grid mode: Manhattan distance between (x, y) cells. OSM mode: network
        shortest-path distance between (lat, lon) node coordinates.
        """
        if self.road_network is not None:
            return self.road_network.route_length_km_latlon(a, b)
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * self.block_km

    def sample_location(self, zone_type):
        """Sample a random location for a given zone type.

        In OSM mode zones do not exist, so a uniformly random network node is
        returned (the deliberately simple choice for home/work/organic spots).
        """
        if self.road_network is not None:
            return self.road_network.sample_node_latlon(self.rng)
        pool = self.zone_index.get(zone_type, [])
        if not pool:
            return (self.rng.randrange(self.size_x), self.rng.randrange(self.size_y))
        return self.rng.choice(pool)

    def sample_location_weighted(self, zone_weights):
        """Sample a location by weighted zone types (random node in OSM mode)."""
        if self.road_network is not None:
            return self.road_network.sample_node_latlon(self.rng)
        zones = []
        weights = []
        for zone_type, weight in zone_weights.items():
            if weight <= 0:
                continue
            if self.zone_index.get(zone_type):
                zones.append(zone_type)
                weights.append(weight)
        if not zones:
            return self.sample_location("mixed")
        chosen_zone = self.rng.choices(zones, weights=weights, k=1)[0]
        return self.sample_location(chosen_zone)

    def community_mode_bias(self, mode):
        """Return a social-practice bias for a mode (SPT)."""
        return self.context.get("community_mode_bias", {}).get(mode, 0.0)
