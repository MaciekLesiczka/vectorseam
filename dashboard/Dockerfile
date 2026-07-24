# VectorSeam calibration dashboard — a dependency-free static server that
# serves the Claude Design dashboard component and synthesizes the rounds.json
# history manifest from the tuner's round-*.json files.
#
# Build context is the repository root:
#   docker build -f dashboard/Dockerfile -t vectorseam-dashboard:local .
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VECTORSEAM_DASHBOARD_STORE=/data/store \
    VECTORSEAM_DASHBOARD_PORT=8080

COPY dashboard/vectorseam_dashboard ./vectorseam_dashboard

EXPOSE 8080

CMD ["python", "-m", "vectorseam_dashboard.server"]
