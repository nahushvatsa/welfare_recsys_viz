"""Latent agent tastes and POI taste tags.

The POI dataset has no cuisine/style field — only a business ``name`` and the
coarse leisure ``category`` the model already maps NAICS onto. This module
recovers a finer "what kind of place is this" signal from the name, and gives
each agent a latent taste that the recommenders **cannot see**.

Why the recommenders cannot see it
----------------------------------
An agent that wants Thai food did not *tell* the platform so. Modelling taste as
a query term would hand the recommender the answer and make ranking trivial.
Instead:

* the tag drives the agent's own **realised activity utility** (a matching place
  is genuinely a better evening), which is ground truth;
* that utility flows into like/dislike feedback, which is the only thing the
  platform observes;
* :meth:`welfare_rs.recommender_systems.RecommenderSystem.record_feedback`
  already accumulates per-token affinities from ``Place.taste_tags``, so the
  platform *infers* the taste over days, by revealed preference.

Relevance is left alone: it compares a POI's **category** tokens against the
subtype's generic query tokens, is near-identical for every candidate in a
subtype, and carries no agent-specific information. Relevance is category fit;
personalization is the only personal channel. Keeping taste tags out of
``Place.keywords`` is deliberate — see the note in ``assign_place_tags``.

Missing data
------------
Only ~45% of real leisure POIs name their type ("Joe's Pizza" does, "Norwind's"
does not), and *which* ones do is biased: pizza and thai self-label, korean and
new-american almost never. Scoring an unlabelled venue as "definitely not thai"
would be a systematic penalty on more than half the catalog, so untagged places
are instead **imputed** a tag drawn from the observed distribution for their
family — the standard missing-at-random treatment, and it preserves the marginal
distribution. Imputation is keyed on a hash of ``place_id`` (never the builtin
``hash()``, which is salted per process and would differ across pool workers),
so a POI's imputed tag is identical across every seed, condition and process.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# ── Vocabulary ───────────────────────────────────────────────────────────────
#
# family -> tag -> substrings matched case-insensitively against the POI name.
# Substrings are matched, not tokens, so "pizzeria" catches "Pizzeria Uno" and
# "bagel" catches "Bagels & Co". Order within a family does not matter: a name
# may earn several tags and keeps all of them.
#
# Terms are chosen to be self-labelling in US business names. Cuisines that are
# rarely named (american, french, "new x") are deliberately absent rather than
# guessed at — the imputation step below covers them without pretending the name
# said something it did not.

TASTE_VOCAB: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "dining": {
        "pizza": ("pizza", "pizzeria", "slice"),
        "sushi": ("sushi", "sashimi", "omakase"),
        "chinese": ("chinese", "szechuan", "sichuan", "hunan", "canton", "dim sum", "wok"),
        "thai": ("thai",),
        "indian": ("indian", "tandoor", "masala", "curry", "biryani"),
        "mexican": ("mexican", "taqueria", "taco", "burrito", "cantina", "tacos"),
        "italian": ("italian", "trattoria", "osteria", "ristorante", "pasta"),
        "japanese": ("japanese", "ramen", "izakaya", "udon", "teriyaki"),
        "korean": ("korean", "kbbq", "bibimbap", "bulgogi"),
        "burger": ("burger", "smashburger"),
        "deli": ("deli", "bagel", "bodega", "sandwich", "hoagie"),
        "bbq": ("bbq", "barbecue", "barbeque", "smokehouse"),
        "seafood": ("seafood", "oyster", "lobster", "crab", "fish market"),
        "vegan": ("vegan", "vegetarian", "plant based", "plant-based"),
        "halal": ("halal", "kosher", "mediterranean", "shawarma", "falafel", "kebab"),
        "steak": ("steak", "chophouse", "grill", "rotisserie"),
        "noodle": ("noodle", "pho", "dumpling"),
        "chicken": ("chicken", "wings", "fried chicken"),
        "bar": ("bar", "tavern", "pub", "cocktail", "lounge", "brewery", "taproom"),
    },
    # Split from dining on purpose. The two never share a leisure subtype
    # (``cafe_friend`` draws only on the ``cafe`` category, the food subtypes
    # never do), so pooling them would let an espresso lover's taste be scored
    # against a steakhouse and vice versa — a match that can never occur.
    "cafe": {
        "coffee": ("coffee", "espresso", "roaster", "roasting", "cafe", "café", "caffe"),
        "bakery": ("bakery", "patisserie", "pastry", "donut", "doughnut", "cake", "bake"),
        "juice": ("juice", "smoothie", "acai", "boba", "bubble tea", "tea house", "teahouse"),
        "dessert": ("ice cream", "gelato", "creamery", "frozen yogurt", "candy", "chocolate"),
        "tea": ("tea", "chai", "matcha"),
    },
    "fitness": {
        "yoga": ("yoga", "pilates", "barre"),
        "gym": ("gym", "fitness", "strength", "weightlift", "crossfit", "athletic club"),
        "cycle": ("cycle", "cycling", "spin studio", "soulcycle"),
        "combat": ("boxing", "muay", "mma", "martial", "karate", "jiu", "judo", "taekwondo"),
        "dance": ("dance", "ballet", "zumba"),
        "swim": ("swim", "aquatic", "natatorium"),
        "climb": ("climb", "bouldering"),
        "tennis": ("tennis", "racquet", "squash", "pickleball"),
    },
    "outdoor": {
        "park": ("park", "commons", "green"),
        "garden": ("garden", "botanic", "arboretum", "conservatory"),
        "trail": ("trail", "greenway", "preserve", "woods", "forest"),
        "waterfront": ("pier", "waterfront", "beach", "marina", "riverside", "esplanade"),
        "playground": ("playground", "recreation", "ballfield", "field"),
        "zoo": ("zoo", "aquarium", "wildlife"),
    },
    "culture": {
        "art": ("art", "gallery", "sculpture", "contemporary"),
        "history": ("history", "historic", "heritage", "memorial", "monument"),
        "science": ("science", "natural history", "planetarium", "discovery"),
        "museum": ("museum",),
    },
    "music": {
        "jazz": ("jazz", "blues", "supper club"),
        "theater": ("theat", "playhouse", "stage", "drama", "opera", "ballet"),
        "concert": ("concert", "arena", "amphitheat", "music hall", "ballroom"),
        "club": ("club", "nightclub", "disco", "lounge"),
        "orchestra": ("orchestra", "symphony", "philharmonic", "chamber"),
    },
}

# Leisure category (as produced by POI_PARAMS' NAICS/text maps) -> taste family.
# A category with no family gets no taste tags and no utility bonus.
#
# The grouping mirrors LEISURE_SUBTYPE_TO_CATEGORIES: every family covers
# exactly the categories that some leisure subtype draws on, so an agent's taste
# in a family is always scoreable against every candidate it will ever see for
# that subtype, and never against candidates from another one.
CATEGORY_TO_FAMILY: Dict[str, str] = {
    "food_takeout": "dining",
    "restaurant": "dining",
    "food": "dining",
    "cafe": "cafe",
    "fitness_studio": "fitness",
    "gym": "fitness",
    "running_route": "fitness",
    "park": "outdoor",
    "museum": "culture",
    "live_music": "music",
    "concert_venue": "music",
    "music_event": "music",
}

FAMILIES: Tuple[str, ...] = tuple(TASTE_VOCAB.keys())


def _name_tags(name: str, family: str) -> Tuple[str, ...]:
    """Tags whose vocabulary appears in ``name``; empty when nothing matches."""
    lowered = (name or "").lower()
    if not lowered:
        return ()
    vocab = TASTE_VOCAB.get(family)
    if not vocab:
        return ()
    return tuple(
        tag for tag, needles in vocab.items() if any(n in lowered for n in needles)
    )


def _stable_unit(place_id: str) -> float:
    """A deterministic float in [0, 1) from a place id.

    ``hashlib`` rather than ``hash()``: the builtin is salted per interpreter, so
    a POI would be imputed differently in each pool worker and matched
    comparisons across conditions would silently break.
    """
    digest = hashlib.md5(str(place_id).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0x100000000


def assign_place_tags(
    places: Sequence[Tuple[str, str, str]],
) -> Dict[str, Tuple[str, ...]]:
    """Map ``place_id -> taste tags`` for ``(place_id, name, category)`` triples.

    Two passes. First every name is matched against its family's vocabulary.
    Then the per-family distribution of *observed* tags is used to impute a
    single tag for each place the first pass could not label, deterministically
    from its id. Both passes are pure functions of the catalog, so the result is
    identical in every process and for every seed and condition.

    These tags are returned separately from ``Place.keywords`` on purpose.
    ``keywords`` feeds the relevance term, which Jaccards them against the
    subtype's generic query tokens — and those tokens are *exactly* the category
    keywords, so a restaurant currently scores a perfect 1.0. Appending "thai"
    would enlarge the union without enlarging the intersection and drop that to
    0.75, i.e. penalise a POI 25% on relevance for the crime of being
    well-labelled. Taste tags therefore live in their own field.
    """
    tagged: Dict[str, Tuple[str, ...]] = {}
    untagged: Dict[str, str] = {}          # place_id -> family
    observed: Dict[str, Dict[str, int]] = {f: {} for f in FAMILIES}

    for place_id, name, category in places:
        family = CATEGORY_TO_FAMILY.get(category)
        if family is None:
            tagged[place_id] = ()
            continue
        tags = _name_tags(name, family)
        if tags:
            tagged[place_id] = tags
            counts = observed[family]
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        else:
            untagged[place_id] = family

    # Impute. Walking the tag names in sorted order makes the cumulative bins
    # reproducible regardless of dict insertion order.
    for place_id, family in untagged.items():
        counts = observed[family]
        if not counts:
            # Nothing in this family self-labelled, so there is no empirical
            # distribution to draw from. Happens with the synthetic fallback
            # catalog, whose "names" are ``restaurant_12``. Impute uniformly
            # from the family's vocabulary instead of leaving every place
            # untagged, which would silently disable tastes for the whole run.
            counts = {tag: 1 for tag in TASTE_VOCAB.get(family, {})}
            if not counts:
                tagged[place_id] = ()
                continue
        names = sorted(counts)
        total = float(sum(counts[t] for t in names))
        target = _stable_unit(place_id) * total
        cumulative = 0.0
        chosen = names[-1]
        for tag in names:
            cumulative += counts[tag]
            if target < cumulative:
                chosen = tag
                break
        tagged[place_id] = (chosen,)

    return tagged


def taste_distribution(
    places: Sequence[Tuple[str, str, str]],
    tags_by_place: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    """``family -> {tag: share}`` over the tagged catalog (shares sum to 1).

    The family comes from each place's own category rather than from a
    tag-name lookup, so two families are free to reuse a tag name without the
    counts bleeding into each other.
    """
    counts: Dict[str, Dict[str, int]] = {f: {} for f in FAMILIES}
    for place_id, _name, category in places:
        family = CATEGORY_TO_FAMILY.get(category)
        if family is None:
            continue
        by_tag = counts[family]
        for tag in tags_by_place.get(place_id, ()):
            by_tag[tag] = by_tag.get(tag, 0) + 1

    out: Dict[str, Dict[str, float]] = {}
    for family, by_tag in counts.items():
        total = float(sum(by_tag.values()))
        out[family] = (
            {tag: by_tag[tag] / total for tag in sorted(by_tag)} if total else {}
        )
    return out


def tastes_per_family(n_tags: int) -> int:
    """How many favourites an agent holds in a family of ``n_tags`` types.

    Scaled to the family's real variety (one favourite per ~8 types, at least
    one) rather than fixed. A flat count misreads the small families: with two
    picks out of the four museum types an agent matches three-quarters of every
    museum it could be shown, which leaves the taste term nearly constant for
    that subtype — the same inertia that makes the relevance term useless.
    """
    return max(1, n_tags // 8)


def sample_agent_tastes(
    distribution: Mapping[str, Mapping[str, float]],
    rng: random.Random,
    beta: float,
    per_family: int = 0,
) -> Dict[str, Tuple[str, ...]]:
    """Draw ``per_family`` distinct latent tastes per family, weight ``share ** beta``.

    ``beta`` tempers how tightly tastes track supply:

    * ``1.0`` — exactly proportional to the observed venue mix.
    * ``0.7`` (default) — the head is flattened. Two known biases both inflate
      it: venues that self-label in their name are over-counted relative to
      those that do not, and small-format types (pizza, deli) need more
      premises to serve the same demand. Shrinking the head corrects both, in
      the direction they err.
    * ``0.0`` — uniform; supply is ignored entirely.

    It also stops the distribution being degenerate. At ``beta = 1`` every
    agent's taste matches whatever is locally abundant, so a taste-aware
    recommender never has to move anybody; below 1, some agents want something
    locally scarce, which is the only way taste produces spatial behaviour.

    ``per_family`` defaults to 0, meaning "scale to the family's variety" via
    :func:`tastes_per_family`. Draws are without replacement, so an agent never
    "likes pizza twice".
    """
    tastes: Dict[str, Tuple[str, ...]] = {}
    for family in FAMILIES:
        shares = distribution.get(family) or {}
        if not shares:
            continue
        tags = [t for t in sorted(shares) if shares[t] > 0.0]
        weights = [shares[t] ** beta for t in tags]
        n_picks = per_family if per_family > 0 else tastes_per_family(len(tags))
        picked: List[str] = []
        for _ in range(n_picks):
            if not tags or sum(weights) <= 0.0:
                break
            choice = rng.choices(tags, weights=weights, k=1)[0]
            index = tags.index(choice)
            tags.pop(index)
            weights.pop(index)
            picked.append(choice)
        if picked:
            tastes[family] = tuple(sorted(picked))
    return tastes


def taste_match(
    agent_tastes: Mapping[str, Sequence[str]], place_tags: Iterable[str]
) -> bool:
    """True when a place carries a tag this agent likes in the same family."""
    wanted = {tag for tags in agent_tastes.values() for tag in tags}
    if not wanted:
        return False
    return any(tag in wanted for tag in place_tags)
