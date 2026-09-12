.PHONY: data-download container-data container-build container-check container-shell container-test \
	container-paper container-gpu-check container-r-fixtures \
	container-r-reference container-r-parallel container-gpu-benchmark

LOCAL_UID := $(shell id -u)
LOCAL_GID := $(shell id -g)
COMPOSE := LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) docker compose -f docker/compose.yaml

container-build:
	$(COMPOSE) build

container-check:
	$(COMPOSE) run --rm check

container-shell:
	$(COMPOSE) run --rm shell

data-download:
	python3 scripts/fetch_validation_data.py

container-data: data-download
	$(COMPOSE) run --rm shell Rscript validation/prepare_inputs.R
	python3 scripts/manifest_prepared_data.py

container-test:
	$(COMPOSE) run --rm test

container-paper:
	$(COMPOSE) run --rm paper

container-gpu-check:
	$(COMPOSE) --profile gpu run --rm gpu

container-r-fixtures:
	$(COMPOSE) run --rm shell Rscript scripts/generate_r_fixtures.R

container-r-reference:
	$(COMPOSE) run --rm shell Rscript bench/run_r.R

container-r-parallel:
	$(COMPOSE) run --rm shell env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
		MKL_NUM_THREADS=1 BENCH_R_WORKERS=12 Rscript bench/run_r_parallel.R

container-gpu-benchmark:
	$(COMPOSE) --profile gpu run --rm gpu \
		python3 bench/bench.py --skip-r --skip-pydeseq2 --device cuda
