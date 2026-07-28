"""Re-score cuDESeq2's output parity against the CACHED R reference, without
re-running R or the PyDESeq2 competitor.

Use this after a change to cuDESeq2 that moves its numbers but not its speed: it
recaptures cuDESeq2 in all three modes, recomputes bench/results/parity.json, and
re-renders bench/results/TABLES.md, while leaving three things untouched --
the R timings and intermediates in bench/cache/, the PyDESeq2 `pj_value` columns
(carried over from the existing parity.json, since PyDESeq2-vs-R does not depend
on our code), and every timing in bench/results/timings.json.

For a full re-benchmark, including timings, run bench/bench.py instead.

Usage: PYTHONPATH=src python bench/refresh_cu_parity.py [--only a,b] [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")

import torch

import run_cu as cu
from bench import DATA, MODES, RES, parity_for_case, render_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="comma-separated case subset")
    args = ap.parse_args()

    cases = sorted(p.name for p in DATA.iterdir() if (p / "meta.json").exists())
    if args.only:
        want = set(args.only.split(","))
        cases = [c for c in cases if c in want]

    old = json.loads((RES / "parity.json").read_text()) if (RES / "parity.json").exists() else {}
    # (case, substep) -> PyDESeq2's score vs R, preserved verbatim.
    pj_prev = {(c, r["substep"]): r["pj_value"]
               for c, rows in old.items() for r in rows if "pj_value" in r}

    parity = {}
    for c in cases:
        print(f"[{c}] cuDESeq2 capture ({'/'.join(MODES)}) ...", flush=True)
        caps = {m: cu.capture_mode(c, m, args.device) for m in MODES}
        rows = parity_for_case(c, caps, pj_cap=None)
        for r in rows:
            if (c, r["substep"]) in pj_prev:
                r["pj_value"] = pj_prev[(c, r["substep"])]
        parity[c] = rows

    if len(cases) < len(old):          # subset run: keep the cases we did not touch
        parity = {**old, **parity}

    (RES / "parity.json").write_text(json.dumps(parity, indent=2))
    t = json.loads((RES / "timings.json").read_text())
    all_cases = sorted(parity)
    (RES / "TABLES.md").write_text(
        render_tables(all_cases, t.get("r", {}), t.get("cu", {}), parity, t.get("pydeseq2")))
    print(f"\nwrote {RES}/parity.json + TABLES.md  (timings.json untouched)")

    fails = [(c, r["substep"], r["value"]) for c, rows in parity.items()
             for r in rows if not r["pass"]]
    n = sum(len(v) for v in parity.values())
    print(f"{n - len(fails)}/{n} substep checks pass")
    for c, s, v in fails:
        print(f"  FAIL {c:20} {s:14} {v:.4g}")


if __name__ == "__main__":
    main()
