# R 4.6.0 / Bioconductor 3.23 are the release pair used for DESeq2 1.52.0.
ARG BIOC_IMAGE=bioconductor/bioconductor:RELEASE_3_23-r-4.6.0
FROM ${BIOC_IMAGE}

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONPATH=/workspace/gpu_deseq/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        poppler-utils \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        texlive-bibtex-extra \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/gpu_deseq

COPY docker/install_r_packages.R /tmp/install_r_packages.R
RUN Rscript /tmp/install_r_packages.R \
    && rm /tmp/install_r_packages.R

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
COPY docker/constraints.txt /tmp/constraints.txt
RUN python3 -m pip install --break-system-packages \
        --index-url "${TORCH_INDEX_URL}" \
        torch==2.6.0 \
    && python3 -m pip install --break-system-packages \
        --constraint /tmp/constraints.txt \
        formulaic \
        gcsfs \
        matplotlib \
        numba \
        numpy \
        pandas \
        pyarrow \
        pytest \
        scipy \
        statsmodels \
        wheel \
    && rm /tmp/constraints.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python3 -m pip install --break-system-packages --no-deps .

# Keep the image useful without a bind mount (tests, fixtures, paper, and
# reference scripts are all present). Large real-data caches are excluded by
# .dockerignore and are mounted from the host for benchmark runs.
COPY . .
RUN python3 docker/check_environment.py \
    && python3 -m pytest tests/test_api.py tests/test_loess.py -q

CMD ["bash"]
