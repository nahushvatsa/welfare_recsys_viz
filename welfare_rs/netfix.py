"""Network environment workarounds for the OSM download path.

Isolated here so the reasons survive: these are about the hosts we talk to and
the box we talk from, not about the model.

Three distinct failures live here, all found the hard way:

1. This box has no routable IPv6 while Overpass publishes AAAA records, and
   Python walks ``getaddrinfo`` sequentially (see :func:`prefer_ipv4`).
2. ``overpass-api.de`` round-robins over several backend IPs and they fail
   *individually* (see :func:`pick_overpass_url`).
3. osmnx's Overpass rate-limiter recurses forever against servers that do no
   slot management (see :func:`configure_overpass`).
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

# Captured once, before anything monkeypatches them, so re-installing our
# wrappers is idempotent instead of building a chain of nested wrappers.
_TRUE_GETADDRINFO = socket.getaddrinfo
_TRUE_GETHOSTBYNAME = socket.gethostbyname

# hostname -> IP address we have positively measured as healthy.
_PINS: dict[str, str] = {}
_FORCE_IPV4: bool | None = None


def _has_ipv6_route() -> bool:
    """True when this host actually has a routable IPv6 path.

    Uses a UDP ``connect``, which only consults the routing table — no packet
    is sent and nothing is contacted.
    """
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        try:
            s.connect(("2001:4860:4860::8888", 53))
            return True
        finally:
            s.close()
    except OSError:
        return False


def _install_dns(force_ipv4: bool) -> None:
    """(Re)install our resolver wrappers over the pristine originals.

    Reinstalling matters: osmnx's ``_http._config_dns`` overwrites
    ``socket.getaddrinfo`` on *every* Overpass request, so our patch is gone
    after the first one. It reads its pinned address from
    ``socket.gethostbyname``, though, which it never replaces — so pinning
    there is what actually survives and steers osmnx.
    """
    def getaddrinfo(host, port, family=0, *args, **kwargs):
        host = _PINS.get(host, host)
        if family == 0 and force_ipv4:
            family = socket.AF_INET
        return _TRUE_GETADDRINFO(host, port, family, *args, **kwargs)

    def gethostbyname(host):
        return _PINS.get(host) or _TRUE_GETHOSTBYNAME(host)

    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname


def prefer_ipv4(force: bool | None = None) -> bool:
    """Resolve hostnames to IPv4 only, when this host has no IPv6 route.

    ``overpass-api.de`` publishes both A and AAAA records. ``curl`` survives
    that on an IPv4-only host because it races both families (Happy Eyeballs,
    RFC 8305); Python does not — ``socket.create_connection`` walks the
    ``getaddrinfo`` list **sequentially**, so every AAAA address burns the full
    connect timeout before IPv4 is ever tried. With osmnx's 180 s timeout that
    surfaces as an intermittent ``ConnectTimeoutError`` partway through a metro
    download, discarding tens of minutes of work (metro.py sets
    ``ox.settings.use_cache = False``, so a failed download retains nothing).

    Filtering AAAA out costs nothing on a host with no IPv6 route — those
    addresses were unusable anyway. Returns True when filtering is active.
    Override the auto-detection with ``WELFARE_RS_FORCE_IPV4=1`` / ``=0``.
    """
    global _FORCE_IPV4
    if force is None:
        if _FORCE_IPV4 is None:
            env = os.environ.get("WELFARE_RS_FORCE_IPV4")
            _FORCE_IPV4 = (
                env.strip() not in ("0", "false", "") if env else not _has_ipv6_route()
            )
        force = _FORCE_IPV4
    else:
        _FORCE_IPV4 = force

    _install_dns(force)
    return force


_CHOSEN_OVERPASS: str | None = None

# Cheap enough that a healthy instance answers in well under a second, but a
# real query rather than /api/status — a struggling instance can serve status
# fine while its query queue is starved, which is exactly how we lost a
# download to a socket that stayed open and moved 943 bytes in 20 seconds.
_PROBE = '[out:json][timeout:25];way(38.895,-77.045,38.900,-77.040)["highway"];out count;'


def _ipv4_addrs(host: str) -> list[str]:
    """Every A record for ``host``, in DNS order, deduped."""
    try:
        infos = _TRUE_GETADDRINFO(host, 443, socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return []
    seen, out = set(), []
    for info in infos:
        ip = info[4][0]
        if ip not in seen:
            seen.add(ip)
            out.append(ip)
    return out


def _probe(url: str, timeout: float) -> float | None:
    """Seconds taken to answer a real query, or None if the endpoint is bad.

    The probe counts ways in a US bounding box and demands a **non-zero**
    count. That is not pedantry: several public instances mirror only their own
    region, and a regional instance answers a US query perfectly — HTTP 200, a
    well-formed Overpass payload, sub-second — with a count of zero.
    ``overpass.osm.ch`` does exactly this (0 ways for Manhattan, Chicago and
    Los Angeles alike). An endpoint check that stops at "200 and parseable"
    accepts it, and every graph built through it comes back empty rather than
    failing, which is far worse than a download that errors out.
    """
    import time

    import requests

    headers = {"User-Agent": "welfare-rs/0.1 (+https://airecsim.cusp.nyu.edu)"}
    # Candidates are BASE urls (osmnx appends "/interpreter"); the probe has to
    # append it too or it would test a path nothing serves.
    probe_url = url.rstrip("/") + "/interpreter"
    t0 = time.time()
    try:
        r = requests.post(probe_url, data={"data": _PROBE},
                          headers=headers, timeout=(10, timeout))
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        ways = int(r.json()["elements"][0]["tags"]["ways"])
    except Exception:
        return None
    if ways <= 0:
        return None
    return time.time() - t0


def pick_overpass_url(candidates=None, timeout: float = 45.0,
                      refresh: bool = False, attempts: int = 3,
                      backoff: float = 20.0) -> str | None:
    """Probe Overpass endpoints and return the fastest healthy one.

    Probing goes down to the **IP address**, not just the hostname, because
    ``overpass-api.de`` is a round-robin over independent backends that fail
    independently: on 2026-07-31 ``162.55.144.139`` (gall) served fine while
    ``65.109.112.52`` (lambert) black-holed every SYN. A hostname-level probe
    is a coin flip — it can succeed on gall and then hand osmnx a run that
    resolves to lambert and dies on a 180 s connect timeout, which is exactly
    what killed the Miami build. So each A record is probed separately and the
    winner is pinned in ``_PINS`` for the rest of the process; osmnx's own
    ``_config_dns`` picks that pin up through ``socket.gethostbyname``.

    An instance counts as healthy only if it returns HTTP 200 *and* a parseable
    Overpass payload — a 200 carrying an error page or an empty regional result
    does not qualify.

    ``refresh`` discards a previously chosen endpoint and probes again. Callers
    should pass it once per metro: a public instance can go from healthy to
    refusing connections within minutes, and a process-wide cache would then
    hand every remaining metro the same dead host.

    The whole sweep is retried ``attempts`` times, because a single sweep is
    not evidence: a healthy kumi.systems measured DEAD (32 s, no response) and
    then 200-in-1.3 s four minutes later while nothing about it had changed.
    One unlucky sweep used to cost a whole metro.

    Returns None only when every endpoint failed every attempt.
    """
    global _CHOSEN_OVERPASS
    if refresh:
        _CHOSEN_OVERPASS = None
    if _CHOSEN_OVERPASS is not None:
        return _CHOSEN_OVERPASS

    import time

    from . import params

    # Self-contained, and re-applied on every call: osmnx clobbers
    # socket.getaddrinfo on each request, so a probe made after the first
    # download would otherwise run with our wrappers already gone.
    prefer_ipv4()

    pinned = params.GEO_PARAMS.get("overpass_url")
    if pinned:
        _CHOSEN_OVERPASS = pinned
        return pinned

    urls = list(candidates or params.GEO_PARAMS.get("overpass_urls") or ())
    # A responder this quick is healthy; stop probing rather than pay a slow
    # mirror's latency just to rank it.
    good_enough = 5.0

    results: list = []
    for attempt in range(max(1, attempts)):
        for url in urls:
            host = urlparse(url).hostname
            if not host:
                continue
            best = None
            for ip in _ipv4_addrs(host):
                _PINS[host] = ip
                elapsed = _probe(url, timeout)
                if elapsed is not None and (best is None or elapsed < best[0]):
                    best = (elapsed, ip)
                if best is not None and best[0] <= good_enough:
                    break
            if best is None:
                _PINS.pop(host, None)
                continue
            _PINS[host] = best[1]
            results.append((best[0], url, host, best[1]))
            if best[0] <= good_enough:
                break
        if results:
            break
        if attempt < attempts - 1:
            time.sleep(backoff)

    if not results:
        return None
    results.sort()
    elapsed, url, host, ip = results[0]
    # Drop pins for the losers so a later refresh re-resolves them freely.
    for _, other_url, other_host, _ip in results[1:]:
        _PINS.pop(other_host, None)
    _PINS[host] = ip
    _CHOSEN_OVERPASS = url
    return url


def overpass_rate_limit(url: str) -> int | None:
    """The server's advertised slot count, or None if it could not be read.

    ``0`` means the instance does no per-client slot management at all.
    """
    import requests

    try:
        r = requests.get(url.rstrip("/") + "/status", timeout=(10, 30),
                         headers={"User-Agent": "welfare-rs/0.1"})
        if r.status_code != 200:
            return None
        for line in r.text.split("\n")[:8]:
            if line.startswith("Rate limit:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def configure_overpass(refresh: bool = True) -> str | None:
    """Point osmnx at a measured-healthy Overpass endpoint and return its URL.

    Also decides whether osmnx's client-side rate limiter may run, because on
    an unmetered instance it **never terminates**. ``_get_overpass_pause``
    reads line 4 of ``/api/status`` and, when its first token is ``Currently``,
    sleeps 5 s and calls itself again. A server advertising ``Rate limit: 0``
    (overpass.kumi.systems) has no "N slots available" line at all, so line 4
    is *always* ``Currently running queries...`` — the recursion can only end
    in a ``RecursionError`` about 80 minutes later, having downloaded nothing.
    That is not a transient outage; it is deterministic, and it is why every
    warm run pointed at kumi has failed.

    Turning the limiter off is safe *and* correct there: the server does no
    slot accounting, and osmnx still honours a 429/504 by pausing 55 s and
    retrying. The limiter is left on whenever the status page cannot be read
    or reports a real limit, since a metered server does need it.
    """
    import osmnx as ox

    from . import params

    url = pick_overpass_url(refresh=refresh)
    if url is None:
        # Do NOT fall through. ``ox.settings.overpass_url`` is process-global
        # and still holds the previous metro's endpoint, so continuing means
        # querying a host we just measured as dead — and osmnx answers an
        # unreachable status page with a flat 60 s pause before spending a
        # 180 s connect timeout on every subquery. That is how DC and Seattle
        # were lost: ~4 minutes per subquery against a host that had already
        # stopped answering SYNs. Failing here costs seconds and names the
        # real cause; warm_metros.py records it and moves to the next metro.
        tried = ", ".join(params.GEO_PARAMS.get("overpass_urls") or ())
        raise RuntimeError(
            f"no healthy Overpass endpoint (tried {tried}) — "
            "refusing to reuse the previous metro's dead endpoint"
        )
    ox.settings.overpass_url = url
    limit = overpass_rate_limit(url)
    ox.settings.overpass_rate_limit = limit != 0
    return url
