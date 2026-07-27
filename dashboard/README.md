# VectorSeam calibration dashboard

A self-contained web dashboard for the VectorSeam tuner's calibration output.
It renders, per cohort, the recommended `ef_search` and its confidence,
recall-vs-latency tradeoff, metrics-over-time charts, and an expandable
round-by-round history.

Two subjects run through the view and are kept apart deliberately. The
**recommended** ef is what a client should apply right now; it is what the
cohort cards, the left-hand chart, and the tradeoff curve show. The
**candidate** ef is what a single round selected on its train split and
tested on its holdout; it is what the right-hand chart and the round history
show. They are the same value whenever a round's candidate clears the
assurance target, and differ when the tuner keeps the previous
recommendation. Every ef on screen is shown with its own confidence, never
the other one's.

A card reads `CARRIED` when the latest round did not replace the
recommendation, with the reason beside the carry timestamp, and shows the
round's candidate underneath when it differs. A history row reads
`NOT APPROVED` when that round's candidate failed to clear the holdout
target. A hollow marker on the left chart means the round kept the previous
recommendation.


## What it reads

The tuner writes, per cohort, under its storage root:

```
<store>/calibrations/<cohort>/latest.json      # newest RoundOutput
<store>/calibrations/<cohort>/round-*.json     # one file per historical round
```

The component fetches `calibrations/<cohort>/latest.json` and a `rounds.json`
history manifest. Static hosting cannot directory-list the `round-*.json`
files, so the server synthesizes `rounds.json` on demand by reading them (newest
`VECTORSEAM_DASHBOARD_MAX_ROUNDS` kept). Each round is passed through whole, so
the component sees the full `RoundOutput` shape (window, coverage, `per_ef`, …).

If no cohorts are configured — or the live store is unreachable — the component
falls back to bundled realistic sample data (`static/sample-data.js`), so the
page always renders.


## Run it

Standalone, against a tuner storage root on the host:

```sh
VECTORSEAM_DASHBOARD_STORE=/path/to/store \
VECTORSEAM_DASHBOARD_COHORTS=superuser \
python -m vectorseam_dashboard.server
# -> http://127.0.0.1:8080
```

Or as a container (build context is the repository root):

```sh
docker build -f dashboard/Dockerfile -t vectorseam-dashboard:local .
docker run --rm -p 8080:8080 \
  -e VECTORSEAM_DASHBOARD_COHORTS=superuser \
  -v /path/to/store:/data/store:ro \
  vectorseam-dashboard:local
```

## Configuration

All via environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VECTORSEAM_DASHBOARD_HOST` | `0.0.0.0` | Bind host |
| `VECTORSEAM_DASHBOARD_PORT` | `8080` | Bind port |
| `VECTORSEAM_DASHBOARD_STORE` | `/data/store` | Tuner storage root (contains `calibrations/`) |
| `VECTORSEAM_DASHBOARD_COHORTS` | *(empty)* | Comma-separated cohorts to show live; empty → sample data |
| `VECTORSEAM_DASHBOARD_BASE_URL` | *(empty)* | Override fetch origin; empty → this page's origin |
| `VECTORSEAM_DASHBOARD_MAX_ROUNDS` | `60` | History points kept per cohort in `rounds.json` |

Note: the component's live path is all-or-nothing per page load — if any listed
cohort has no published `latest.json`, the whole view drops to sample data. List
only cohorts that are actually publishing.
