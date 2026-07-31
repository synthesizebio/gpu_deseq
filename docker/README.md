# Reproducible container

The image combines the two environments needed by the paper:

- R 4.6.0, Bioconductor 3.23, DESeq2 1.52.0, apeglm 1.34.0;
- Python 3.11 or newer, PyTorch 2.6.0 with CUDA 12.4, Triton 3.2.0, and
  the exact direct dependency versions in `constraints.txt`;
- the LaTeX and PDF tools needed to build `paper/main.pdf`.

The base tag is the official Bioconductor R-4.6.0 release image. PyTorch uses
the official `cu124` wheel index. `check_environment.py` fails if any
manuscript-critical package version changes.

## Build and check

```bash
make container-build
make container-check
make container-test
make container-paper
```

`container-test` excludes `test_bulk_v4_holdout_benchmark.py` because that test
depends on a predictions file in another repository. All self-contained
repository tests run.

The repository is bind-mounted into the container. Compose runs with the host
UID and GID so generated fixtures, tables, figures, and PDFs remain editable by
the host user.

## Regenerate the R reference

Prepared real-data inputs must exist in `validation/data/`; that directory is
intentionally excluded from the image because the GTEx input is large. With the
inputs present on the host:

```bash
make container-r-fixtures
make container-r-reference
```

Both commands write through the bind mount.

## GPU check and benchmark

Install the NVIDIA Container Toolkit on the host and configure Docker to use
the NVIDIA runtime. Then run:

```bash
make container-gpu-check
make container-gpu-benchmark
```

The GPU benchmark reuses the R reference in `bench/cache/` and reruns all six
GPU cases. This is the command needed to replace the pending GTEx timing.

The CUDA libraries are supplied by the PyTorch wheel. The host supplies only a
compatible NVIDIA driver and the GPU device through the container runtime.

## Direct Compose use

The Make targets pass the current host UID/GID automatically. For direct use:

```bash
LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" \
  docker compose -f docker/compose.yaml run --rm shell
```

To use an image mirror or an immutable digest, override `BIOC_IMAGE`:

```bash
docker build \
  --build-arg BIOC_IMAGE='bioconductor/bioconductor:RELEASE_3_23-r-4.6.0@sha256:…' \
  -f docker/Dockerfile -t gpu-deseq:bioc3.23-cu124 .
```
