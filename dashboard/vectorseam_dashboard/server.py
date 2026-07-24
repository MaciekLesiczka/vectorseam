"""Static server for the VectorSeam calibration dashboard.

Serves the Claude Design dashboard component (``index.dc.html`` + its
``support.js`` runtime, vendored React/ReactDOM/Babel, and ``sample-data.js``)
together with the read-only calibration store the tuner writes under
``STORE_ROOT/calibrations/<cohort>/``.

The tuner emits one ``latest.json`` plus discrete ``round-*.json`` files per
cohort; static hosting cannot directory-list the latter, so this server
synthesizes the ``rounds.json`` history manifest the dashboard fetches by
reading those files on demand.

Standard library only — the container needs no third-party dependencies. The
dashboard is configured entirely through environment variables:

    VECTORSEAM_DASHBOARD_HOST     bind host (default 0.0.0.0)
    VECTORSEAM_DASHBOARD_PORT     bind port (default 8080)
    VECTORSEAM_DASHBOARD_STORE    calibration store root (default /data/store)
    VECTORSEAM_DASHBOARD_COHORTS  comma-separated cohort names to render live;
                                  empty -> the dashboard uses bundled sample data
    VECTORSEAM_DASHBOARD_MAX_ROUNDS  history points kept per cohort (default 60)
    VECTORSEAM_DASHBOARD_BASE_URL    baseUrl for live fetches (default "" = same origin)
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re

STATIC_ROOT = Path(__file__).resolve().parent / "static"
DC_DOCUMENT = "index.dc.html"

# The dashboard component references React/ReactDOM/Babel from unpkg. The
# vendored copies are wired in by config.js; support.js loads config.js only
# because we inject it ahead of its own <script src="./support.js"> tag.
_SUPPORT_TAG = '<script src="./support.js"></script>'
_INJECTED_TAGS = '<script src="./config.js"></script>\n' + _SUPPORT_TAG

_COHORT_SEGMENT = re.compile(r"\A[A-Za-z0-9._-]+\Z")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class Config:
    """Server configuration resolved from the environment once at startup."""

    def __init__(self, environ: "os._Environ[str] | dict[str, str]") -> None:
        self.host = environ.get("VECTORSEAM_DASHBOARD_HOST", "0.0.0.0")
        self.port = int(environ.get("VECTORSEAM_DASHBOARD_PORT", "8080"))
        self.store_root = Path(
            environ.get("VECTORSEAM_DASHBOARD_STORE", "/data/store")
        ).resolve()
        self.max_rounds = int(
            environ.get("VECTORSEAM_DASHBOARD_MAX_ROUNDS", "60")
        )
        self.base_url = environ.get("VECTORSEAM_DASHBOARD_BASE_URL", "")
        raw_cohorts = environ.get("VECTORSEAM_DASHBOARD_COHORTS", "")
        self.cohorts = [c.strip() for c in raw_cohorts.split(",") if c.strip()]

    def client_config_json(self) -> str:
        """The window.VECTORSEAM_CONFIG payload injected into config.js.

        Empty when no cohorts are configured, which leaves VECTORSEAM_CONFIG
        unset so the dashboard renders its bundled sample data.
        """
        if not self.cohorts:
            return ""
        return json.dumps({"baseUrl": self.base_url, "cohorts": self.cohorts})


def build_rounds_manifest(cohort_dir: Path, max_rounds: int) -> dict[str, object]:
    """Concatenates round-*.json documents into a history manifest.

    Rounds sort by filename, which the tuner names with a UTC timestamp, so
    lexical order is chronological. Unreadable or malformed files are skipped
    rather than failing the whole manifest. Each round is returned whole: the
    dashboard consumes the full RoundOutput shape (window, coverage, per_ef…).
    """
    rounds: list[object] = []
    for path in sorted(cohort_dir.glob("round-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rounds.append(payload)
    return rounds[-max_rounds:] if max_rounds > 0 else rounds


def make_handler(config: Config) -> type[BaseHTTPRequestHandler]:
    """Builds a request handler bound to the resolved configuration."""

    calibrations_root = (config.store_root / "calibrations").resolve()

    def safe_cohort_dir(cohort: str) -> Path | None:
        if not _COHORT_SEGMENT.match(cohort):
            return None
        cohort_dir = (calibrations_root / cohort).resolve()
        if calibrations_root not in cohort_dir.parents:
            return None
        return cohort_dir

    class Handler(BaseHTTPRequestHandler):
        server_version = "vectorseam-dashboard/1.0"

        def do_GET(self) -> None:  # noqa: N802 - required name
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html", "/" + DC_DOCUMENT):
                self._send_dc_document()
                return
            if path == "/config.js":
                self._send_config_js()
                return
            if path == "/healthz":
                self._send_bytes(b'{"status":"ok"}\n', ".json")
                return

            match = re.fullmatch(r"/calibrations/([^/]+)/([^/]+)", path)
            if match:
                self._send_calibration(match.group(1), match.group(2))
                return

            self._send_static(path.lstrip("/"))

        # -- dynamic routes ------------------------------------------------

        def _send_dc_document(self) -> None:
            """Serves the component, injecting config.js before support.js."""
            source = (STATIC_ROOT / DC_DOCUMENT).read_text(encoding="utf-8")
            html = source.replace(_SUPPORT_TAG, _INJECTED_TAGS, 1)
            self._send_bytes(html.encode("utf-8"), ".html")

        def _send_config_js(self) -> None:
            """Serves config.js with the live wiring substituted from env."""
            source = (STATIC_ROOT / "config.js").read_text(encoding="utf-8")
            payload = config.client_config_json()
            # Substitute the placeholder token (still a valid JS string literal
            # when empty, so the file stays usable opened directly too).
            js = source.replace('"__DASHBOARD_CONFIG__"', json.dumps(payload), 1)
            self._send_bytes(js.encode("utf-8"), ".js")

        def _send_calibration(self, cohort: str, name: str) -> None:
            cohort_dir = safe_cohort_dir(cohort)
            if cohort_dir is None:
                self._send_status(404)
                return
            if name == "rounds.json":
                manifest = build_rounds_manifest(cohort_dir, config.max_rounds)
                self._send_bytes(json.dumps(manifest).encode("utf-8"), ".json")
                return
            if name == "latest.json":
                try:
                    body = (cohort_dir / "latest.json").read_bytes()
                except OSError:
                    self._send_status(404)
                    return
                self._send_bytes(body, ".json")
                return
            self._send_status(404)

        # -- static files --------------------------------------------------

        def _send_static(self, relative: str) -> None:
            target = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT != target and STATIC_ROOT not in target.parents:
                self._send_status(404)
                return
            try:
                body = target.read_bytes()
            except (OSError, IsADirectoryError):
                self._send_status(404)
                return
            self._send_bytes(body, target.suffix)

        # -- helpers -------------------------------------------------------

        def _send_bytes(self, body: bytes, suffix: str) -> None:
            content_type = _CONTENT_TYPES.get(
                suffix, "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_status(self, code: int) -> None:
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            print(
                f"dashboard {self.address_string()} {fmt % args}", flush=True
            )

    return Handler


def main() -> None:
    config = Config(os.environ)
    handler = make_handler(config)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    cohorts = ", ".join(config.cohorts) if config.cohorts else "(sample data)"
    print(
        f"dashboard serving on {config.host}:{config.port} "
        f"store={config.store_root} cohorts={cohorts}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
