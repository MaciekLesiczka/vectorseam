"""Client for the effective-recommendation HTTP API.

``RecommendationClient`` reads ``GET /v1/ef-search/<cohort>`` from the
recommendation server and caches each cohort's ``ef_search`` for a TTL.
Lookups are lazy: the first read of a cohort after its entry expires 
performs one HTTP request.
"""

from __future__ import annotations

from http import HTTPStatus
import time
import urllib.error
import urllib.parse
import urllib.request

# Mirrors MIN_EF_SEARCH/MAX_EF_SEARCH in vectorseam-core.
MIN_EF_SEARCH = 1
MAX_EF_SEARCH = 1000

DEFAULT_PORT = 7738
DEFAULT_MAX_COHORTS = 10_000


class RecommendationClient:
    """Caching reader for effective per-cohort ``ef_search`` recommendations.

    The client is safe to read from multiple threads. Concurrent misses for
    one cohort may each issue a request; the server coalesces them.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        default_ef_search: int = 100,
        ttl_seconds: float = 60.0,
        timeout_seconds: float = 1.0,
        max_cohorts: int = DEFAULT_MAX_COHORTS,
    ) -> None:
        """Initializes a recommendation client.

        Args:
            host: Recommendation server host.
            port: Recommendation server port.
            default_ef_search: Value served until a cohort is read once.
            ttl_seconds: Lifetime of a cached value, successful or not.
            timeout_seconds: Deadline for one HTTP request.
            max_cohorts: Maximum number of cohorts ever cached.

        Raises:
            ValueError: An argument is outside its accepted range.
        """
        if not host:
            raise ValueError("host must be non-empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not MIN_EF_SEARCH <= default_ef_search <= MAX_EF_SEARCH:
            raise ValueError(
                f"default_ef_search must be between {MIN_EF_SEARCH} "
                f"and {MAX_EF_SEARCH}"
            )
        if ttl_seconds < 0.0:
            raise ValueError("ttl_seconds must not be negative")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if max_cohorts < 1:
            raise ValueError("max_cohorts must be at least 1")

        self._base_url = f"http://{host}:{port}/v1/ef-search/"
        self._default_ef_search = default_ef_search
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._max_cohorts = max_cohorts
        # Entries are never evicted. Once max_cohorts distinct cohorts have
        # been seen, further cohorts are refused rather than admitted: an
        # eviction policy would thrash to a near-total miss rate under a
        # runaway caller, and every miss is one request to the collector.
        self._cache: dict[str, tuple[int, float]] = {}

    def ef_search(self, cohort: str) -> int:
        """Returns the cached or freshly fetched ``ef_search`` for a cohort.

        A cohort first seen after ``max_cohorts`` are already cached is served
        the default without a lookup, and is never cached.

        Raises:
            ValueError: The recommendation server rejects the cohort as invalid.
        """
        cached = self._cache.get(cohort)
        now = time.monotonic()
        if cached is not None and now < cached[1]:
            return cached[0]
        if cached is None and len(self._cache) >= self._max_cohorts:
            return self._default_ef_search

        fallback = cached[0] if cached is not None else self._default_ef_search
        ef_search = self._fetch(cohort)
        if ef_search is None:
            ef_search = fallback
        self._cache[cohort] = (ef_search, now + self._ttl_seconds)
        return ef_search

    def _fetch(self, cohort: str) -> int | None:
        """Reads one recommendation, or None when it is unusable."""
        url = self._base_url + urllib.parse.quote(cohort, safe="/")
        try:
            with urllib.request.urlopen(
                url, timeout=self._timeout_seconds
            ) as response:
                # The endpoint answers with a plain-text integer.
                body = response.read(32)
        except urllib.error.HTTPError as error:
            # A non-2xx response is raised, and still holds an open body.
            error.close()
            if error.code == HTTPStatus.BAD_REQUEST:
                raise ValueError(f"invalid cohort: {cohort!r}") from error
            return None
        except (urllib.error.URLError, OSError, TimeoutError):
            return None

        try:
            ef_search = int(body)
        except ValueError:
            return None
        if not MIN_EF_SEARCH <= ef_search <= MAX_EF_SEARCH:
            return None
        return ef_search
