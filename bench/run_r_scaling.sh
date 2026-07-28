#!/usr/bin/env bash
# =============================================================================
# run_r_scaling.sh — R DESeq2 multi-core scaling benchmark (standalone)
#
# Measures how R DESeq2 1.30.1 scales across CPU cores on the six real
# benchmark datasets, so the paper can report the GPU speedup against R given a
# genuinely large CPU budget instead of only a single-threaded baseline.
#
# Designed to be run on a FRESH machine with no editing:
#
#     git clone <repo> && cd gpu_deseq
#     bash bench/run_r_scaling.sh --data-from user@a100-host:/home/max_synthesize_bio/gpu_deseq
#
# It bootstraps the pinned R stack, obtains the datasets, runs the sweep, and
# writes bench/results/R_SCALING.md + r_scaling.jsonl.
#
# -----------------------------------------------------------------------------
# WHAT IT MEASURES, AND WHY IT IS SHAPED THIS WAY
#
# In DESeq2 1.30.1 the `parallel=TRUE` argument exists only on DESeq(),
# results() and lfcShrink(). estimateDispersions() and nbinomWaldTest() have no
# parallel argument at all: DESeq2:::DESeqParallel fuses size factors +
# dispersion + Wald into a single bplapply over gene chunks. A per-substep
# parallel breakdown therefore does not exist in this version, so the sweep is
# measured as three coarse blocks:
#
#   deseq_fit  = DESeq()      -> size factors + dispersion + Wald  (parallel)
#   results    = results()                                          (parallel)
#   lfc_shrink = lfcShrink(type="apeglm")                           (parallel)
#
# Two tracks are recorded:
#
#   Track A (serial, fine-grained) replicates bench/run_r.R's exact five-substep
#     call sequence at one thread, so these numbers are directly comparable to
#     the existing R column in bench/results/TABLES.md.
#
#   Track B (swept, coarse) runs the SAME three blocks at every worker count
#     including w=1. The w=1 point is the correct serial baseline for the
#     scaling curve, because only the worker count varies along it. Do not
#     compute parallel speedup against Track A -- the call paths differ.
#
# DESeq() is called with minReplicatesForReplace=Inf, which disables the
# replaceOutliers refit. This is the fair setting: cuDESeq2 does not implement
# that refit (see the paper's Limitations), so leaving it on would charge R for
# work the GPU never does and inflate the reported speedup.
#
# BLAS threading is pinned to 1 everywhere. Otherwise a multithreaded OpenBLAS
# would silently invalidate the "R single-threaded" baseline, and would
# oversubscribe the box at high worker counts. The BLAS in use is recorded in
# the output header, because absolute serial timings are only comparable across
# machines when the BLAS matches.
#
# This script is timing-only. It deliberately does not export intermediates or
# check parity -- the parity artifacts already exist on the GPU host and do not
# depend on core count.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------- defaults ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOCKER_IMG="bioconductor/bioconductor_docker:RELEASE_3_12"
RLIB_DOCKER="$REPO_ROOT/bench/.rlib-bioc312"
OUT_DIR="$REPO_ROOT/bench/results"
DATA_DIR="$REPO_ROOT/validation/data"

WORKERS=""            # default computed from physical core count
REPS_SMALL=3
REPS_LARGE=1
LARGE_N=100           # n_samples above this uses REPS_LARGE
ONLY=""
DATA_FROM=""
FETCH_DATA=0
R_MODE="auto"         # auto | native | docker
ANY_DESEQ2=0          # accept a native DESeq2 whose version is not 1.30.1
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash bench/run_r_scaling.sh [options]

  --workers 1,2,4,8,16,32,44   Worker counts to sweep (default: powers of two
                               up to the physical core count, plus that count).
  --reps-small N               Reps per point for n_samples <= 100 (default 3).
  --reps-large N               Reps per point for n_samples >  100 (default 1).
  --only case[,case...]        Subset of datasets (default: all present).
  --data-from HOST:PATH        rsync validation/data/ from a machine that has it
                               (e.g. user@a100-host:/path/to/gpu_deseq).
  --fetch-data                 Regenerate datasets from Bioconductor instead.
                               NOTE: the GTEx case downloads a 1.37 GB recount2
                               RSE and needs the pasilla + airway data packages.
  --r-mode auto|native|docker  How to get R 4.0 / DESeq2 1.30.1 (default auto:
                               native if already present, else docker).
  --allow-any-deseq2           Accept whatever DESeq2 the native R provides
                               instead of requiring exactly 1.30.1. Use when a
                               host cannot run the pinned Bioc 3.12 image (no
                               docker, no disk). The measured version is
                               recorded in the output header -- timings taken
                               this way are NOT directly comparable to the
                               1.30.1 R column in TABLES.md, since the DESeq2
                               version varies alongside the hardware.
  --out DIR                    Output directory (default bench/results).
  --dry-run                    Print the plan and exit.
  -h, --help                   This message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)     WORKERS="$2"; shift 2 ;;
    --reps-small)  REPS_SMALL="$2"; shift 2 ;;
    --reps-large)  REPS_LARGE="$2"; shift 2 ;;
    --only)        ONLY="$2"; shift 2 ;;
    --data-from)   DATA_FROM="$2"; shift 2 ;;
    --fetch-data)  FETCH_DATA=1; shift ;;
    --r-mode)      R_MODE="$2"; shift 2 ;;
    --allow-any-deseq2) ANY_DESEQ2=1; shift ;;
    --out)         OUT_DIR="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------- machine inventory ---
PHYS_CORES="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l || echo 0)"
[[ "$PHYS_CORES" -gt 0 ]] || PHYS_CORES="$(nproc)"
LOGICAL_CPUS="$(nproc)"
CPU_MODEL="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
MEM_GB="$(free -g | awk '/^Mem:/{print $2}')"

if [[ -z "$WORKERS" ]]; then
  # Powers of two up to the physical core count, then the core count itself.
  # Physical, not logical: DESeq2's per-gene IRLS is dense FP64 arithmetic with
  # nothing for a hyperthread sibling to hide, so logical counts mostly measure
  # port contention rather than added throughput.
  ws=(); w=1
  while [[ $w -lt $PHYS_CORES ]]; do ws+=("$w"); w=$((w*2)); done
  ws+=("$PHYS_CORES")
  WORKERS="$(IFS=,; echo "${ws[*]}")"
fi

log "host: $CPU_MODEL"
log "cores: $PHYS_CORES physical / $LOGICAL_CPUS logical, ${MEM_GB} GB RAM"
log "worker sweep: $WORKERS"

# --------------------------------------------------------------- R backend ---
# The default gate insists on DESeq2 1.30.1 so that "native" always means the
# same software as the docker fallback -- otherwise a host with a newer DESeq2
# would silently produce numbers that are not comparable to the reference R
# column. --allow-any-deseq2 waives the version pin (but not the requirement
# that DESeq2 and apeglm exist) for hosts that cannot run the pinned image.
have_native_r() {
  command -v Rscript >/dev/null 2>&1 || return 1
  ANY_DESEQ2="$ANY_DESEQ2" Rscript -e '
    .libPaths(c(Sys.getenv("R_DESEQ2_LIB", unset="~/R/library"), .libPaths()))
    any_ver <- Sys.getenv("ANY_DESEQ2") == "1"
    ok <- requireNamespace("DESeq2", quietly=TRUE) &&
          requireNamespace("apeglm", quietly=TRUE) &&
          (any_ver || as.character(packageVersion("DESeq2")) == "1.30.1")
    q(status = if (ok) 0 else 1)' >/dev/null 2>&1
}

if [[ "$R_MODE" == "auto" ]]; then
  if have_native_r; then R_MODE="native"; else R_MODE="docker"; fi
fi
log "R backend: $R_MODE"

# RUN_R <script.R> [args...]  -- executes an R script against the pinned stack.
if [[ "$R_MODE" == "native" ]]; then
  have_native_r || die "--r-mode native but $([[ $ANY_DESEQ2 -eq 1 ]] && echo 'DESeq2 + apeglm' || echo 'DESeq2 1.30.1 + apeglm') not found (set R_DESEQ2_LIB, or pass --allow-any-deseq2)"
  if [[ $ANY_DESEQ2 -eq 1 ]]; then
    NATIVE_DESEQ2="$(R_DESEQ2_LIB="${R_DESEQ2_LIB:-$HOME/R/library}" Rscript -e \
      '.libPaths(c(Sys.getenv("R_DESEQ2_LIB"), .libPaths())); cat(as.character(packageVersion("DESeq2")))' 2>/dev/null)"
    [[ "$NATIVE_DESEQ2" == "1.30.1" ]] || \
      log "WARNING: using DESeq2 $NATIVE_DESEQ2, not the reference 1.30.1 -- timings are not directly comparable to the R column in bench/results/TABLES.md"
  fi
  RUN_R() {
    R_DESEQ2_LIB="${R_DESEQ2_LIB:-$HOME/R/library}" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    BENCH_WORKERS="$WORKERS" BENCH_REPS_SMALL="$REPS_SMALL" \
    BENCH_REPS_LARGE="$REPS_LARGE" BENCH_LARGE_N="$LARGE_N" \
    BENCH_ONLY="$ONLY" BENCH_OUT="$OUT_DIR" BENCH_DATA="$DATA_DIR" \
    BENCH_HOST_DESC="$CPU_MODEL | ${PHYS_CORES}p/${LOGICAL_CPUS}l cores | ${MEM_GB}GB" \
      Rscript "$@"
  }
else
  command -v docker >/dev/null 2>&1 || die "docker not found. Install it (apt-get install -y docker.io) or use --r-mode native."
  docker info >/dev/null 2>&1 || die "docker present but not usable by this user. Try: sudo usermod -aG docker \$USER && newgrp docker"
  mkdir -p "$RLIB_DOCKER"
  log "pulling $DOCKER_IMG (pins R 4.0 + Bioconductor 3.12 => DESeq2 1.30.1, apeglm 1.12.0)"
  docker pull -q "$DOCKER_IMG" >/dev/null

  DOCKER_R() {
    docker run --rm \
      -v "$REPO_ROOT:/work" -v "$RLIB_DOCKER:/rlib" -w /work \
      -e R_LIBS_USER=/rlib \
      -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
      -e BENCH_WORKERS="$WORKERS" -e BENCH_REPS_SMALL="$REPS_SMALL" \
      -e BENCH_REPS_LARGE="$REPS_LARGE" -e BENCH_LARGE_N="$LARGE_N" \
      -e BENCH_ONLY="$ONLY" -e BENCH_OUT="/work/${OUT_DIR#$REPO_ROOT/}" \
      -e BENCH_DATA="/work/validation/data" \
      -e BENCH_HOST_DESC="$CPU_MODEL | ${PHYS_CORES}p/${LOGICAL_CPUS}l cores | ${MEM_GB}GB (docker)" \
      "$DOCKER_IMG" "$@"
  }
  RUN_R() { DOCKER_R Rscript "$@"; }

  if [[ $DRY_RUN -eq 0 ]]; then
    log "ensuring DESeq2 1.30.1 + apeglm in $RLIB_DOCKER (first run compiles; ~10-20 min)"
    DOCKER_R Rscript -e "
      options(Ncpus = $PHYS_CORES)
      need <- c('DESeq2','apeglm')
      miss <- need[!vapply(need, requireNamespace, logical(1), quietly=TRUE)]
      if (length(miss)) BiocManager::install(miss, ask=FALSE, update=FALSE)
      cat('DESeq2', as.character(packageVersion('DESeq2')),
          '/ apeglm', as.character(packageVersion('apeglm')), '\n')"
  fi
fi

# ------------------------------------------------------------------- data ----
mkdir -p "$OUT_DIR"
# -L so a validation/data symlinked onto a bigger disk is still discovered (the
# datasets are ~2 GB with GTEx, which often does not fit beside the repo).
present_cases() { find -L "$DATA_DIR" -mindepth 2 -maxdepth 2 -name meta.json -printf '%h\n' 2>/dev/null | xargs -r -n1 basename | sort; }

if [[ -n "$DATA_FROM" ]]; then
  command -v rsync >/dev/null 2>&1 || die "rsync not found (apt-get install -y rsync)"
  log "syncing datasets from $DATA_FROM"
  mkdir -p "$DATA_DIR"
  rsync -a --info=progress2 "${DATA_FROM%/}/validation/data/" "$DATA_DIR/"
elif [[ $FETCH_DATA -eq 1 ]]; then
  log "regenerating datasets from Bioconductor (GTEx step downloads 1.37 GB)"
  RUN_R -e "
    options(Ncpus = $PHYS_CORES)
    need <- c('pasilla','airway','SummarizedExperiment')
    miss <- need[!vapply(need, requireNamespace, logical(1), quietly=TRUE)]
    if (length(miss)) BiocManager::install(miss, ask=FALSE, update=FALSE)"
  RUN_R validation/fetch_and_reference.R
  RUN_R validation/prepare_gtex.R || log "WARNING: GTEx preparation failed; continuing without that case"
fi

CASES="$(present_cases || true)"
if [[ -z "$CASES" ]]; then
  die "no datasets under $DATA_DIR.
validation/data/ is gitignored, so a fresh clone does not contain it. Either:
  --data-from user@a100-host:/home/max_synthesize_bio/gpu_deseq   (rsync, ~85 MB, fastest)
  --fetch-data                                                    (regenerate; 1.37 GB GTEx download)"
fi
log "datasets found: $(echo "$CASES" | tr '\n' ' ')"
[[ -n "$ONLY" ]] && log "restricting to: $ONLY"

if [[ $DRY_RUN -eq 1 ]]; then
  log "dry run complete; nothing executed"
  exit 0
fi

# ------------------------------------------------- embedded R benchmark -------
BENCH_R="$(mktemp "${TMPDIR:-/tmp}/r_scaling_XXXXXX.R")"
trap 'rm -f "$BENCH_R"' EXIT

cat >"$BENCH_R" <<'EOF_R'
# Embedded by bench/run_r_scaling.sh. Configuration arrives via env vars.
.libPaths(c(Sys.getenv("R_DESEQ2_LIB", unset = "~/R/library"), .libPaths()))
suppressMessages({library(DESeq2); library(apeglm); library(BiocParallel)})

DATA      <- Sys.getenv("BENCH_DATA")
OUT       <- Sys.getenv("BENCH_OUT")
WORKERS   <- as.integer(strsplit(Sys.getenv("BENCH_WORKERS"), ",")[[1]])
REPS_SM   <- as.integer(Sys.getenv("BENCH_REPS_SMALL", "3"))
REPS_LG   <- as.integer(Sys.getenv("BENCH_REPS_LARGE", "1"))
LARGE_N   <- as.integer(Sys.getenv("BENCH_LARGE_N", "100"))
ONLY      <- Sys.getenv("BENCH_ONLY", "")
HOST_DESC <- Sys.getenv("BENCH_HOST_DESC", "unknown")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

JSONL <- file.path(OUT, "r_scaling.jsonl")
MD    <- file.path(OUT, "R_SCALING.md")

cases <- list.dirs(DATA, recursive = FALSE, full.names = FALSE)
cases <- Filter(function(c) file.exists(file.path(DATA, c, "meta.json")), cases)
if (nzchar(ONLY)) cases <- intersect(cases, trimws(strsplit(ONLY, ",")[[1]]))
if (!length(cases)) stop("no cases to run")

# Same minimal flat-JSON reader as bench/run_r.R, so metadata is read identically.
read_meta <- function(p) {
  txt <- paste(readLines(p), collapse = " ")
  get <- function(k) { m <- regmatches(txt, regexpr(sprintf('"%s"\\s*:\\s*"?([^",}]+)', k), txt)); sub(sprintf('"%s"\\s*:\\s*"?', k), "", m) }
  list(design = get("design"), factor = get("factor"), ref = get("ref"), nonref = get("nonref"),
       coef = get("coef"), n_samples = as.integer(get("n_samples")))
}
secs  <- function(t0) as.numeric(Sys.time() - t0, units = "secs")
emit  <- function(rec) cat(paste0(rec, "\n"), file = JSONL, append = TRUE)

blas <- tryCatch(sessionInfo()$BLAS, error = function(e) NA)
if (is.null(blas) || is.na(blas)) blas <- "unknown (likely reference)"

cat(sprintf("R scaling benchmark\n  host: %s\n  R %s | DESeq2 %s | apeglm %s\n  BLAS: %s\n  workers: %s\n\n",
            HOST_DESC, getRversion(), packageVersion("DESeq2"), packageVersion("apeglm"),
            blas, paste(WORKERS, collapse = ",")))

# Fresh JSONL each invocation; the markdown table is rendered from it at the end.
if (file.exists(JSONL)) unlink(JSONL)
emit(sprintf('{"kind":"meta","host":"%s","r":"%s","deseq2":"%s","apeglm":"%s","blas":"%s","workers":"%s","utc":"%s"}',
             HOST_DESC, getRversion(), packageVersion("DESeq2"), packageVersion("apeglm"),
             gsub('"', "", blas), paste(WORKERS, collapse = ","),
             format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")))

for (case in cases) {
  d <- file.path(DATA, case)
  meta <- read_meta(file.path(d, "meta.json"))
  cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1)); storage.mode(cts) <- "integer"
  cd  <- read.csv(file.path(d, "coldata.csv"), row.names = 1)
  cd[[meta$factor]] <- relevel(factor(cd[[meta$factor]]), ref = meta$ref)
  reps <- if (meta$n_samples > LARGE_N) REPS_LG else REPS_SM
  mk <- function() DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))
  P <- ncol(model.matrix(as.formula(meta$design), cd))

  cat(sprintf("== %s  (n=%d, genes=%d, P=%d, design=%s, reps=%d)\n",
              case, ncol(cts), nrow(cts), P, meta$design, reps))

  # ---- Track A: serial, fine-grained. Mirrors bench/run_r.R exactly so these
  # ---- numbers slot into the existing R column of bench/results/TABLES.md.
  tt <- list(normalization = c(), dispersion = c(), glm_fit = c(), significance = c(), lfc_shrink = c())
  for (i in seq_len(reps)) {
    dds <- mk()
    t0 <- Sys.time(); dds <- estimateSizeFactors(dds);               tt$normalization <- c(tt$normalization, secs(t0))
    t0 <- Sys.time(); dds <- estimateDispersions(dds, quiet = TRUE); tt$dispersion    <- c(tt$dispersion,    secs(t0))
    t0 <- Sys.time(); dds <- nbinomWaldTest(dds);                    tt$glm_fit       <- c(tt$glm_fit,       secs(t0))
    t0 <- Sys.time(); res <- results(dds, name = meta$coef);         tt$significance  <- c(tt$significance,  secs(t0))
    t0 <- Sys.time(); sh  <- lfcShrink(dds, coef = meta$coef, type = "apeglm", quiet = TRUE)
                                                                     tt$lfc_shrink    <- c(tt$lfc_shrink,    secs(t0))
  }
  fine <- lapply(tt, function(x) median(x) * 1000)
  fine$total <- Reduce(`+`, fine)
  emit(sprintf('{"kind":"serial_fine","case":"%s","n":%d,"genes":%d,"P":%d,"reps":%d,"normalization":%.3f,"dispersion":%.3f,"glm_fit":%.3f,"significance":%.3f,"lfc_shrink":%.3f,"total":%.3f}',
               case, ncol(cts), nrow(cts), P, reps, fine$normalization, fine$dispersion,
               fine$glm_fit, fine$significance, fine$lfc_shrink, fine$total))
  cat(sprintf("   serial fine   norm=%.0f disp=%.0f glm=%.0f sig=%.0f shrink=%.0f  total=%.0f ms\n",
              fine$normalization, fine$dispersion, fine$glm_fit, fine$significance,
              fine$lfc_shrink, fine$total))

  # ---- Track B: coarse blocks, swept over worker count. Identical call path at
  # ---- every w (including w=1), so the curve is internally consistent.
  # ---- minReplicatesForReplace=Inf disables the replaceOutliers refit, which
  # ---- cuDESeq2 does not implement -- charging R for it would inflate speedup.
  for (w in WORKERS) {
    par <- w > 1
    bp  <- if (par) MulticoreParam(workers = w) else SerialParam()
    tf <- tr <- ts <- c()
    ok <- TRUE
    for (i in seq_len(reps)) {
      r <- tryCatch({
        dds <- mk()
        t0 <- Sys.time()
        dds <- DESeq(dds, quiet = TRUE, minReplicatesForReplace = Inf,
                     parallel = par, BPPARAM = bp)
        a <- secs(t0)
        t0 <- Sys.time()
        res <- results(dds, name = meta$coef, parallel = par, BPPARAM = bp)
        b <- secs(t0)
        t0 <- Sys.time()
        sh <- lfcShrink(dds, coef = meta$coef, type = "apeglm", quiet = TRUE,
                        parallel = par, BPPARAM = bp)
        c(a, b, secs(t0))
      }, error = function(e) { cat(sprintf("   w=%-3d FAILED: %s\n", w, conditionMessage(e))); NULL })
      if (is.null(r)) { ok <- FALSE; break }
      tf <- c(tf, r[1]); tr <- c(tr, r[2]); ts <- c(ts, r[3])
    }
    if (!ok) {
      emit(sprintf('{"kind":"sweep","case":"%s","workers":%d,"failed":true}', case, w))
      next
    }
    fit <- median(tf) * 1000; rs <- median(tr) * 1000; shr <- median(ts) * 1000
    tot <- fit + rs + shr
    emit(sprintf('{"kind":"sweep","case":"%s","workers":%d,"reps":%d,"deseq_fit":%.3f,"results":%.3f,"lfc_shrink":%.3f,"total":%.3f}',
                 case, w, reps, fit, rs, shr, tot))
    cat(sprintf("   w=%-3d  deseq_fit=%-9.0f results=%-7.0f lfc_shrink=%-8.0f total=%.0f ms\n",
                w, fit, rs, shr, tot))
    if (par) { tryCatch(bpstop(bp), error = function(e) invisible(NULL)); gc(FALSE) }
  }
  cat("\n")
}

# ------------------------------------------------------------ render tables ---
lines <- readLines(JSONL)
# Quoted values are read up to the closing quote, not to the next comma, so
# comma-bearing strings (the worker list) survive intact.
fld <- function(s, k, num = TRUE) {
  if (num) {
    m <- regmatches(s, regexpr(sprintf('"%s":\\s*-?[0-9.]+([eE][-+]?[0-9]+)?', k), s))
    if (!length(m)) return(NA_real_)
    return(as.numeric(sub(sprintf('"%s":\\s*', k), "", m)))
  }
  m <- regmatches(s, regexpr(sprintf('"%s":\\s*"[^"]*"', k), s))
  if (!length(m)) return(NA_character_)
  sub('"$', "", sub(sprintf('"%s":\\s*"', k), "", m))
}
kind  <- vapply(lines, fld, character(1), "kind", FALSE, USE.NAMES = FALSE)
metaL <- lines[kind == "meta"][1]
fineL <- lines[kind == "serial_fine"]
sweeL <- lines[kind == "sweep"]

fmt <- function(x) if (is.na(x)) "--" else formatC(round(x), format = "d", big.mark = "")
out <- c()
add <- function(...) out <<- c(out, paste0(...))

add("# R DESeq2 multi-core scaling")
add("")
add("Generated by `bench/run_r_scaling.sh`. Timing only -- no parity checks.")
add("")
add(sprintf("- host: `%s`", fld(metaL, "host", FALSE)))
add(sprintf("- R %s, DESeq2 %s, apeglm %s", fld(metaL, "r", FALSE),
            fld(metaL, "deseq2", FALSE), fld(metaL, "apeglm", FALSE)))
add(sprintf("- BLAS: `%s` (all BLAS/OMP thread counts pinned to 1)", fld(metaL, "blas", FALSE)))
add(sprintf("- worker sweep: %s", fld(metaL, "workers", FALSE)))
add(sprintf("- generated: %s", fld(metaL, "utc", FALSE)))
add("")
add("## Table A -- serial, fine-grained (median ms, 1 thread)")
add("")
add("Replicates `bench/run_r.R`'s five-substep call sequence, so these are")
add("directly comparable to the R column of `bench/results/TABLES.md`. Compare")
add("against the previous host to see how much of any change is clock speed.")
add("")
add("| dataset | n | genes | P | normalization | dispersion | glm_fit | significance | lfc_shrink | total |")
add("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for (s in fineL) {
  add(sprintf("| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s |",
              fld(s, "case", FALSE), fld(s, "n"), fld(s, "genes"), fld(s, "P"),
              fmt(fld(s, "normalization")), fmt(fld(s, "dispersion")), fmt(fld(s, "glm_fit")),
              fmt(fld(s, "significance")), fmt(fld(s, "lfc_shrink")), fmt(fld(s, "total"))))
}

sw_case <- vapply(sweeL, fld, character(1), "case", FALSE, USE.NAMES = FALSE)
sw_w    <- vapply(sweeL, fld, numeric(1),   "workers", USE.NAMES = FALSE)
sw_tot  <- vapply(sweeL, fld, numeric(1),   "total",   USE.NAMES = FALSE)
ws      <- sort(unique(sw_w))
ucases  <- unique(sw_case)

add("")
add("## Table B -- total pipeline vs worker count (median ms)")
add("")
add("`DESeq(minReplicatesForReplace=Inf)` + `results()` + `lfcShrink(apeglm)`,")
add("identical call path at every worker count. The `w=1` column is the correct")
add("serial baseline for parallel speedup -- not Table A, whose call path differs.")
add("")
add(paste0("| dataset | ", paste(sprintf("w=%d", ws), collapse = " | "), " | best | speedup @best | parallel efficiency @best |"))
add(paste0("|---|", paste(rep("--:", length(ws) + 3), collapse = "|"), "|"))
for (cs in ucases) {
  row <- vapply(ws, function(w) { v <- sw_tot[sw_case == cs & sw_w == w]; if (length(v)) v[1] else NA_real_ }, numeric(1))
  base <- row[ws == 1]
  bi <- which.min(row)
  spd <- if (length(base) && !is.na(base) && !is.na(row[bi])) base / row[bi] else NA_real_
  eff <- if (!is.na(spd)) 100 * spd / ws[bi] else NA_real_
  add(sprintf("| %s | %s | w=%d | %s | %s |", cs,
              paste(vapply(row, fmt, character(1)), collapse = " | "), ws[bi],
              if (is.na(spd)) "--" else sprintf("%.1fx", spd),
              if (is.na(eff)) "--" else sprintf("%.0f%%", eff)))
}

add("")
add("## Table C -- per-block scaling (median ms)")
add("")
add("Which block stops scaling. DESeq2 1.30.1 has no `parallel` argument on")
add("`estimateDispersions`/`nbinomWaldTest`; `DESeqParallel` fuses size factors +")
add("dispersion + Wald into one `bplapply`, so `deseq_fit` is the finest")
add("parallel-resolvable unit for those three substeps.")
add("")
add("| dataset | block | ", paste(sprintf("w=%d", ws), collapse = " | "), " | speedup w1->best |")
add(paste0("|---|---|", paste(rep("--:", length(ws) + 1), collapse = "|"), "|"))
for (cs in ucases) {
  for (blk in c("deseq_fit", "results", "lfc_shrink")) {
    row <- vapply(ws, function(w) {
      s <- sweeL[sw_case == cs & sw_w == w]
      if (length(s)) fld(s[1], blk) else NA_real_
    }, numeric(1))
    base <- row[ws == 1]
    spd <- if (length(base) && !is.na(base)) base / min(row, na.rm = TRUE) else NA_real_
    add(sprintf("| %s | %s | %s | %s |", cs, blk,
                paste(vapply(row, fmt, character(1)), collapse = " | "),
                if (is.na(spd)) "--" else sprintf("%.1fx", spd)))
  }
}

add("")
add("## Reading these numbers")
add("")
add("- Small-*n* datasets may get **slower** with more workers: forking and")
add("  serialising a chunked `DESeqDataSet` costs more than the per-gene fits it")
add("  saves. That is a real result, not a misconfiguration.")
add("- Parallel efficiency well under 100% at the best worker count is the point")
add("  of this table: `DESeq2` leaves size-factor estimation, the dispersion trend")
add("  fit, and parts of `results` serial, so the curve plateaus regardless of how")
add("  many cores are available.")
add("- Absolute serial times are only comparable across machines when CPU clock")
add("  and BLAS match. Re-measure both the serial and parallel points on any new")
add("  host rather than mixing a new parallel number against an old baseline.")
add("")

writeLines(out, MD)
cat(sprintf("\nWrote:\n  %s\n  %s\n", MD, JSONL))
EOF_R

# ------------------------------------------------------------------- run -----
log "starting sweep (this can take 30-90 min depending on core count)"
START=$(date +%s)

if [[ "$R_MODE" == "docker" ]]; then
  # The temp script lives outside the repo, so mount it in explicitly.
  docker run --rm \
    -v "$REPO_ROOT:/work" -v "$RLIB_DOCKER:/rlib" -v "$BENCH_R:/bench.R:ro" -w /work \
    -e R_LIBS_USER=/rlib \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -e BENCH_WORKERS="$WORKERS" -e BENCH_REPS_SMALL="$REPS_SMALL" \
    -e BENCH_REPS_LARGE="$REPS_LARGE" -e BENCH_LARGE_N="$LARGE_N" \
    -e BENCH_ONLY="$ONLY" -e BENCH_OUT="/work/${OUT_DIR#$REPO_ROOT/}" \
    -e BENCH_DATA="/work/validation/data" \
    -e BENCH_HOST_DESC="$CPU_MODEL | ${PHYS_CORES}p/${LOGICAL_CPUS}l cores | ${MEM_GB}GB (docker)" \
    "$DOCKER_IMG" Rscript /bench.R
else
  RUN_R "$BENCH_R"
fi

log "done in $(( ($(date +%s) - START) / 60 )) min"
log "results: $OUT_DIR/R_SCALING.md  and  $OUT_DIR/r_scaling.jsonl"
