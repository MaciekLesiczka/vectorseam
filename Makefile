.PHONY: setup postgres ann-recall-latency-download ann-recall-latency-load ann-recall-latency-embed ann-recall-latency-pg-load ann-recall-latency-ground-truth ann-recall-latency-sweep ann-recall-latency-analyze all-ann-recall-latency test-rust test-python test fmt clean

CARGO  ?= cargo
UV     ?= uv
ANN_RECALL_LATENCY_COMPOSE := python/ann-recall-latency/docker-compose.yml
ANN_RECALL_LATENCY_POSTGRES_DATA := $(CURDIR)/python/ann-recall-latency/data/postgres
ANN_RECALL_LATENCY_DOCKER_COMPOSE := ANN_RECALL_LATENCY_POSTGRES_DATA="$(ANN_RECALL_LATENCY_POSTGRES_DATA)" docker compose -f $(ANN_RECALL_LATENCY_COMPOSE)

postgres:
	$(ANN_RECALL_LATENCY_DOCKER_COMPOSE) up -d postgres

setup:
	$(UV) venv --allow-existing
	$(UV) pip install -e .

ann-recall-latency-download: setup
	$(UV) run python python/ann-recall-latency/download.py --dataset all

ann-recall-latency-load: setup
	$(UV) run python python/ann-recall-latency/load.py --dataset all

ann-recall-latency-embed: setup
	$(UV) run python python/ann-recall-latency/embed.py --dataset all

ann-recall-latency-pg-load: postgres setup
	$(UV) run python python/ann-recall-latency/pg_load.py --dataset all

ann-recall-latency-ground-truth: setup
	$(UV) run python python/ann-recall-latency/ground_truth.py --dataset all

ann-recall-latency-sweep: postgres setup
	$(UV) run python python/ann-recall-latency/sweep.py --dataset all

ann-recall-latency-analyze: setup
	$(UV) run python python/ann-recall-latency/analyze.py

all-ann-recall-latency: setup postgres
	$(UV) run python python/ann-recall-latency/download.py --dataset all
	$(UV) run python python/ann-recall-latency/load.py --dataset all
	$(UV) run python python/ann-recall-latency/embed.py --dataset all
	$(UV) run python python/ann-recall-latency/pg_load.py --dataset all
	$(UV) run python python/ann-recall-latency/ground_truth.py --dataset all
	$(UV) run python python/ann-recall-latency/sweep.py --dataset all
	$(UV) run python python/ann-recall-latency/analyze.py

test-rust:
	$(CARGO) test --workspace

test-python: setup
	$(UV) run python -m compileall python
	$(UV) run python -m unittest discover -s python/vectorseam/tests

test: test-rust test-python

fmt:
	$(CARGO) fmt --all

clean:
	rm -rf .venv
	$(CARGO) clean
	$(ANN_RECALL_LATENCY_DOCKER_COMPOSE) down
