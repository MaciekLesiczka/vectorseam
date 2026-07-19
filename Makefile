.PHONY: setup postgres ann-recall-latency-download ann-recall-latency-load ann-recall-latency-embed ann-recall-latency-pg-load ann-recall-latency-ground-truth ann-recall-latency-sweep ann-recall-latency-analyze all-ann-recall-latency seam-postgres-up seam-postgres-down seam-f-pg-fixture seam-f-pg-tests seam-anchor seam-anchor-tests seam-f-pg-harness test-seam-f-agg bench-seam-phase-b build-rust test-rust test-rust-msrv lint-rust doc-rust test-python bench-python bench-python-frame bench-python-vector-capture bench-python-report bench-python-frame-report bench-python-vector-capture-report test fmt clean

CARGO  ?= cargo
UV     ?= uv
RUST_MSRV ?= 1.85.0
RUST_MSRV_TARGET_DIR ?= $(CURDIR)/target/msrv-$(RUST_MSRV)
PYTHON_FRAME_BENCH_JSON ?= .benchmarks/frame.json
PYTHON_VECTOR_CAPTURE_BENCH_JSON ?= .benchmarks/vector_capture.json
ANN_RECALL_LATENCY_COMPOSE := python/ann-recall-latency/docker-compose.yml
ANN_RECALL_LATENCY_POSTGRES_DATA := $(CURDIR)/python/ann-recall-latency/data/postgres
ANN_RECALL_LATENCY_DOCKER_COMPOSE := ANN_RECALL_LATENCY_POSTGRES_DATA="$(ANN_RECALL_LATENCY_POSTGRES_DATA)" docker compose -f $(ANN_RECALL_LATENCY_COMPOSE)
SEAM_COMPOSE := tests/seam/docker-compose.yml
SEAM_PG_PORT ?= 55432
SEAM_DATABASE_URL ?= postgresql://postgres:password@localhost:$(SEAM_PG_PORT)/postgres
SEAM_F_PG_ROOT ?= $(CURDIR)/target/seam-fixtures/f-pg
SEAM_DOCKER_COMPOSE := SEAM_PG_PORT="$(SEAM_PG_PORT)" docker compose -f $(SEAM_COMPOSE)

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

seam-postgres-up:
	$(SEAM_DOCKER_COMPOSE) up -d --wait postgres

seam-postgres-down:
	$(SEAM_DOCKER_COMPOSE) down --volumes

seam-f-pg-fixture: seam-postgres-up
	DATABASE_URL="$(SEAM_DATABASE_URL)" SEAM_F_PG_ROOT="$(SEAM_F_PG_ROOT)" \
		$(CARGO) run --release -p seam --example f_pg_fixture

seam-anchor: seam-f-pg-fixture setup
	rm -f "$(SEAM_F_PG_ROOT)/anchor/comparison.json"
	$(UV) run python -m seam_harness.anchor \
		--fixture-root "$(SEAM_F_PG_ROOT)" \
		--dsn "$(SEAM_DATABASE_URL)"
	test -s "$(SEAM_F_PG_ROOT)/anchor/comparison.json"

seam-f-pg-tests: seam-f-pg-fixture
	SEAM_REQUIRE_F_PG=1 SEAM_PG_PORT="$(SEAM_PG_PORT)" \
		SEAM_DATABASE_URL="$(SEAM_DATABASE_URL)" SEAM_TEST_PG_PASSWORD=password \
		$(CARGO) test -p seam --lib -- --ignored

seam-anchor-tests: seam-anchor
	SEAM_REQUIRE_F_PG=1 SEAM_PG_PORT="$(SEAM_PG_PORT)" \
		SEAM_F_PG_ROOT="$(SEAM_F_PG_ROOT)" \
		SEAM_TEST_PG_PASSWORD=password \
		$(CARGO) test -p seam --test acceptance_a_anchor -- --ignored

seam-f-pg-harness: seam-f-pg-tests seam-anchor-tests

test-seam-f-agg:
	$(CARGO) test -p seam \
		--lib \
		--test f_agg_builders \
		--test acceptance_b_estimator \
		--test acceptance_c_durability \
		--test estimator_properties

bench-seam-phase-b:
	$(CARGO) bench -p seam --bench phase_b -- --noplot

build-rust:
	$(CARGO) build --workspace

test-rust:
	$(CARGO) test --workspace

# Resolve every compiler executable through rustup: `rustup run cargo` can
# otherwise spawn a newer `rustc` or `rustdoc` found first on PATH. The
# dedicated target is cleaned so dependencies are always compiled by MSRV.
test-rust-msrv:
	@set -eu; \
		msrv_cargo="$$(rustup which --toolchain $(RUST_MSRV) cargo)"; \
		msrv_rustc="$$(rustup which --toolchain $(RUST_MSRV) rustc)"; \
		msrv_rustdoc="$$(rustup which --toolchain $(RUST_MSRV) rustdoc)"; \
		test "$$("$$msrv_cargo" --version | cut -d ' ' -f 2)" = "$(RUST_MSRV)"; \
		test "$$("$$msrv_rustc" --version | cut -d ' ' -f 2)" = "$(RUST_MSRV)"; \
		test "$$("$$msrv_rustdoc" --version | cut -d ' ' -f 2)" = "$(RUST_MSRV)"; \
		echo "MSRV cargo: $$("$$msrv_cargo" --version)"; \
		echo "MSRV rustc: $$("$$msrv_rustc" --version)"; \
		echo "MSRV rustdoc: $$("$$msrv_rustdoc" --version)"; \
		CARGO_TARGET_DIR="$(RUST_MSRV_TARGET_DIR)" "$$msrv_cargo" clean; \
		RUSTC="$$msrv_rustc" RUSTDOC="$$msrv_rustdoc" \
			CARGO_TARGET_DIR="$(RUST_MSRV_TARGET_DIR)" \
			"$$msrv_cargo" check --workspace --all-targets --locked; \
		RUSTC="$$msrv_rustc" RUSTDOC="$$msrv_rustdoc" \
			CARGO_TARGET_DIR="$(RUST_MSRV_TARGET_DIR)" \
			"$$msrv_cargo" test --workspace --locked

lint-rust:
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets -- -D warnings

doc-rust:
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --workspace --no-deps

test-python: setup
	$(UV) run python -m compileall python
	$(UV) run python -m unittest discover -s python/vectorseam/tests

bench-python: bench-python-frame bench-python-vector-capture

bench-python-frame:
	mkdir -p .benchmarks
	$(UV) run --extra bench python benchmarks/bench_frame.py --output $(PYTHON_FRAME_BENCH_JSON)

bench-python-vector-capture:
	mkdir -p .benchmarks
	$(UV) run --extra bench python benchmarks/bench_vector_capture.py --output $(PYTHON_VECTOR_CAPTURE_BENCH_JSON)

bench-python-report: bench-python-frame-report bench-python-vector-capture-report

bench-python-frame-report:
	@if [ -f "$(PYTHON_FRAME_BENCH_JSON)" ]; then \
		$(UV) run --extra bench python -m pyperf stats "$(PYTHON_FRAME_BENCH_JSON)"; \
	else \
		echo "Missing $(PYTHON_FRAME_BENCH_JSON); run make bench-python-frame"; \
	fi

bench-python-vector-capture-report:
	@if [ -f "$(PYTHON_VECTOR_CAPTURE_BENCH_JSON)" ]; then \
		$(UV) run --extra bench python -m pyperf stats "$(PYTHON_VECTOR_CAPTURE_BENCH_JSON)"; \
	else \
		echo "Missing $(PYTHON_VECTOR_CAPTURE_BENCH_JSON); run make bench-python-vector-capture"; \
	fi

test: test-rust test-python

fmt:
	$(CARGO) fmt --all

clean:
	rm -rf .venv
	$(CARGO) clean
	$(ANN_RECALL_LATENCY_DOCKER_COMPOSE) down
	$(SEAM_DOCKER_COMPOSE) down --volumes
