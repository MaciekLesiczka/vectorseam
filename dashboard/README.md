# VectorSeam calibration dashboard

A self-contained web dashboard for the VectorSeam tuner's calibration output.
It renders, per cohort, the train-selected and currently effective
`ef_search`, training and holdout confidence, recall-vs-latency tradeoff,
metrics-over-time charts, and an expandable round-by-round history. A hollow
ef marker means the round carried a prior effective recommendation.


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
