#!/usr/bin/env bash
# paper_build — recompile the cuDESeq2 whitepaper (paper/main.tex -> paper/main.pdf)
#
# Runs the standard pdflatex/bibtex/pdflatex/pdflatex sequence so citations and
# cross-references resolve. Works from any directory. On failure, prints the
# offending LaTeX errors and exits non-zero.
#
# Usage:
#   scripts/paper_build.sh            # full build (default)
#   scripts/paper_build.sh --quick    # single pdflatex pass (fast, skips bibtex)
#   scripts/paper_build.sh --figures  # regenerate figures first, then full build
#   scripts/paper_build.sh --clean    # remove build artifacts, then full build
set -euo pipefail

# --- locate the paper directory relative to this script ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
JOB="main"

QUICK=0
FIGURES=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --quick)   QUICK=1 ;;
    --figures) FIGURES=1 ;;
    --clean)   CLEAN=1 ;;
    -h|--help)
      # print the leading comment block (skip the shebang), stop at first code line
      awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) echo "paper_build: unknown option '$arg' (try --help)" >&2; exit 2 ;;
  esac
done

command -v pdflatex >/dev/null 2>&1 || { echo "paper_build: pdflatex not found on PATH" >&2; exit 1; }
[ -f "$PAPER_DIR/$JOB.tex" ] || { echo "paper_build: $PAPER_DIR/$JOB.tex not found" >&2; exit 1; }
cd "$PAPER_DIR"

# run a latex/bibtex command quietly; on failure, surface the errors and abort.
run_tex() {
  local label="$1"; shift
  echo ">> $label"
  if ! "$@" >/dev/null 2>&1; then
    echo "paper_build: $label FAILED —" >&2
    if [ "$1" = bibtex ]; then
      grep -B1 -A2 -E 'error|I was expecting|missing|skipping' "$JOB.blg" 2>/dev/null | head -40 >&2 || true
    else
      grep -A3 -E '^(!|.*Error)' "$JOB.log" 2>/dev/null | head -40 >&2 || true
    fi
    exit 1
  fi
}

if [ "$CLEAN" = 1 ]; then
  echo ">> clean"
  rm -f "$JOB".{aux,bbl,blg,log,out,toc,fls,fdb_latexmk,synctex.gz} "$JOB.pdf"
fi

if [ "$FIGURES" = 1 ]; then
  for fig in make_schematic_figure.py make_optimizations_figure.py; do
    [ -f "$fig" ] || continue
    echo ">> figure: $fig"
    python "$fig" >/dev/null 2>&1 || { echo "paper_build: $fig FAILED" >&2; exit 1; }
  done
fi

PDFLATEX=(pdflatex -interaction=nonstopmode -halt-on-error "$JOB")

if [ "$QUICK" = 1 ]; then
  run_tex "pdflatex (quick)" "${PDFLATEX[@]}"
else
  run_tex "pdflatex (1/3)" "${PDFLATEX[@]}"
  if command -v bibtex >/dev/null 2>&1; then
    run_tex "bibtex" bibtex "$JOB"
  else
    echo "paper_build: bibtex not found — skipping (citations may be stale)" >&2
  fi
  run_tex "pdflatex (2/3)" "${PDFLATEX[@]}"
  run_tex "pdflatex (3/3)" "${PDFLATEX[@]}"
fi

# --- report ----------------------------------------------------------------
if command -v pdfinfo >/dev/null 2>&1; then
  PAGES="$(pdfinfo "$JOB.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')"
  echo "OK: $PAPER_DIR/$JOB.pdf (${PAGES:-?} pages)"
else
  echo "OK: $PAPER_DIR/$JOB.pdf"
fi

# surface remaining undefined refs/citations as warnings (non-fatal)
if grep -qE 'Warning: (Citation|Reference|There were undefined)' "$JOB.log" 2>/dev/null; then
  echo "note: unresolved references/citations remain:" >&2
  grep -E 'Warning: (Citation|Reference)' "$JOB.log" | sort -u | head -20 >&2 || true
fi
