"""Check manuscript numbers against the committed benchmark artifacts.

This audit is deliberately read-only.  It checks the current DESeq2 reference
version, the two manuscript tables derived from the real-data run, and the
headline speedup and parity summaries.  It also
requires every ``\fact{}`` marker to have a named source class.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import statistics

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper/main.tex").read_text()
TEXN = re.sub(r"\s+", " ", TEX)
PARITY = json.loads((ROOT / "bench/results/reference_parity.json").read_text())
TIMING = json.loads((ROOT / "bench/results/timings.json").read_text())
PARALLEL_R = json.loads(
    (ROOT / "bench/results/r_parallel_a100_12worker.json").read_text()
)
MINMU = json.loads(
    (ROOT / "bench/results/minmu_counterfactual.json").read_text()
)

fails: list[str] = []
checks = 0


def check(label: str, condition: bool, detail: object = "") -> None:
    global checks
    checks += 1
    if not condition:
        fails.append(f"{label}: {detail}")


def close(a: float, b: float, rtol: float = 0.015) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-15)


def table(label: str) -> str:
    match = re.search(
        rf"\\label\{{{re.escape(label)}\}}.*?\\end\{{tabular\}}", TEX, re.S
    )
    if not match:
        raise RuntimeError(f"table {label!r} not found")
    return match.group()


# Current R reference provenance.
for case, record in TIMING["r"].items():
    check(f"{case}: staged R pipeline",
          record.get("pipeline") == "staged_standard_wald_no_duplicate_dispersion")
    check(f"{case}: R glm_fit scope",
          record.get("glm_fit_scope") == "nbinomWaldTest_plus_outlier_replacement_refit")
    check(f"{case}: staged output equivalence",
          record.get("output_equivalent_to_DESeq") is True)
for case, record in PARITY["cases"].items():
    check(f"{case}: DESeq2 version", record["deseq2_version"] == "1.52.0")
    check(f"{case}: R version", record["r_version"] == "4.6.0")
    check(f"{case}: apeglm version", record["apeglm_version"] == "1.34.0")
check("standard reference pipeline",
      PARITY["pipeline"] == "standard_wald_with_outlier_refit")
check("parallel R version", PARALLEL_R["provenance"]["r_version"] == "R version 4.6.0 (2026-04-24)")
check("parallel DESeq2 version", PARALLEL_R["provenance"]["deseq2_version"] == "1.52.0")
check("parallel apeglm version", PARALLEL_R["provenance"]["apeglm_version"] == "1.34.0")
check("parallel worker count", PARALLEL_R["provenance"]["workers"] == 12)
check("parallel benchmark covers six real designs", len(PARALLEL_R["cases"]) == 6)


# Table 2: medians and least-favorable values across six designs.
metrics: dict[str, list[float]] = {}
directions: dict[str, bool] = {}
all_rows = []
for record in PARITY["cases"].values():
    for row in record["metrics"]:
        metrics.setdefault(row["substep"], []).append(row["value"])
        directions[row["substep"]] = row["higher_better"]
        all_rows.append(row)
check("30 real-data checks", len(all_rows) == 30, len(all_rows))
check("all real-data checks pass", all(row["pass"] for row in all_rows))
check("real-data check count", len(all_rows) == 30)
check("dispersion worst is 0.44 percent",
      round(100 * max(metrics["dispersion"]), 2) == .44)
check("dispersion worst raw value",
      close(max(metrics["dispersion"]), 4.4e-3, .01))
airway_metrics = {
    row["substep"]: row["value"] for row in PARITY["cases"]["airway"]["metrics"]
}
check("airway dispersion p95", close(airway_metrics["dispersion"], 4.5e-4, .01))
check("airway LFC p95", close(airway_metrics["glm_fit"], 3.0e-5, .02))
check("airway exact called set", airway_metrics["significance"] == 1.0)
check("all shrinkage correlations at least 0.997",
      min(metrics["lfc_shrink"]) >= .997)
validation_reports = [
    json.loads((ROOT / f"validation/results/{case}.json").read_text())
    for case in PARITY["cases"]
]
check("all shrinkage p95 absolute errors at most 3.4e-3",
      max(row["shrunk_lfc_p95_abs"] for row in validation_reports) <= 3.4e-3)

parity_table = table("tab:parity")
for substep, values in metrics.items():
    row = next(
        line for line in parity_table.splitlines()
        if line.startswith(substep.replace("_", r"\_") + " &")
    )
    cells = [cell.strip() for cell in row.split("&")]
    shown_median = float(cells[-2])
    shown_worst = float(cells[-1].split(r"\\")[0])
    actual_median = statistics.median(values)
    actual_worst = min(values) if directions[substep] else max(values)
    check(f"Table 2 {substep} median", close(shown_median, actual_median, .06),
          (shown_median, actual_median))
    check(f"Table 2 {substep} worst", close(shown_worst, actual_worst, .06),
          (shown_worst, actual_worst))


# Tables 3 and 4: current R and A100 cells for all six designs.
labels = [
    ("pasilla", "pasilla"),
    ("pasilla_2fac", r"pasilla\_2fac"),
    ("airway_dex", r"airway\_dex"),
    ("airway_cell", r"airway\_cell"),
    ("airway", "airway"),
    ("gtex_blood_muscle", "gtex"),
]
timing_table = table("tab:realtiming")
substep_table = table("tab:realsubstep")
speedups = []
for case, label in labels:
    row = next(line for line in timing_table.splitlines()
               if line.strip().startswith(label + " "))
    values = [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", row)]
    parallel_record = PARALLEL_R["cases"][case]
    check(f"Table 3 {case}: R one-worker seconds",
          round(values[2], 1) == round(parallel_record["serial"]["median_ms"] / 1000, 1),
          values)
    check(f"Table 3 {case}: R 12-worker seconds",
          round(values[3], 1) == round(parallel_record["parallel"]["median_ms"] / 1000, 1),
          values)
    shown_gpu = values[4:7]
    actual_gpu = [TIMING["cu"][case][mode]["total"]
                  for mode in ("eager", "graph", "triton")]
    check(f"Table 3 {case}: GPU totals",
          all(round(a) == round(b) for a, b in zip(shown_gpu, actual_gpu)),
          (shown_gpu, actual_gpu))
    speedups.append(min(parallel_record["serial"]["median_ms"],
                         parallel_record["parallel"]["median_ms"]) / min(actual_gpu))

    block_start = substep_table.index(label + " &")
    next_mid = substep_table.find(r"\midrule", block_start)
    block = substep_table[block_start:next_mid if next_mid >= 0 else None]
    for stage in ("dispersion", "glm_fit", "significance", "lfc_shrink", "total"):
        stage_label = stage.replace("_", r"\_")
        line = next(line for line in block.splitlines() if stage_label in line)
        shown = ([int(x) for x in re.findall(r"\\textbf\{(\d+)\}", line)]
                 if stage == "total"
                 else [int(x) for x in re.findall(r"&\s*(\d+)", line)])
        actual = [round(TIMING["r"][case][stage])] + [
            round(TIMING["cu"][case][mode][stage])
            for mode in ("eager", "graph", "triton")
        ]
        check(f"Table 4 {case}/{stage}", shown == actual, (shown, actual))

check("Table 2 staged-chain description",
      "times an output-equivalent staged call path" in TEXN)
check("top-N claim is limited to plotted values",
      "Each of the 14 logarithmically spaced $N$ values shown" in TEXN)
check("abstract labels dispersion metric",
      "final dispersion has a median p95 relative error" in TEXN)

headline = re.findall(r"\\fact\{2\.9--13\.4\$\\times\$\}", TEX)
check("headline range appears three times", len(headline) == 3, len(headline))
check("headline minimum", round(min(speedups), 1) == 2.9, min(speedups))
check("headline maximum", round(max(speedups), 1) == 13.4, max(speedups))
check("cuDESeq2 normalization at most 4 ms",
      max(TIMING["cu"][case][mode]["normalization"]
          for case, _ in labels for mode in ("eager", "graph", "triton")) <= 4.0)
check("real-data Triton dispersion speedups and endpoints",
      round(TIMING["cu"]["pasilla_2fac"]["eager"]["dispersion"] /
            TIMING["cu"]["pasilla_2fac"]["triton"]["dispersion"], 1) == 8.7
      and round(TIMING["cu"]["airway"]["eager"]["dispersion"] /
                TIMING["cu"]["airway"]["triton"]["dispersion"], 1) == 2.7
      and round(TIMING["cu"]["pasilla_2fac"]["eager"]["dispersion"]) == 730
      and round(TIMING["cu"]["pasilla_2fac"]["triton"]["dispersion"]) == 83
      and round(TIMING["cu"]["airway"]["eager"]["dispersion"]) == 1586
      and round(TIMING["cu"]["airway"]["triton"]["dispersion"]) == 583)


# Called-set claims are derived from the current standard-pipeline CSVs.
jaccards = {}
for case, _ in labels:
    r_path = ROOT / f"bench/cache/{case}/r_results.csv"
    o_path = ROOT / f"bench/cache/{case}/cu_reference_results.csv"
    r = pd.read_csv(r_path)
    o = pd.read_csv(o_path)
    rset = set(r.index[r["padj"] < .05])
    oset = set(o.index[o["padj"] < .05])
    jaccards[case] = len(rset & oset) / len(rset | oset)
check("four exact called sets", sum(v == 1 for v in jaccards.values()) == 4,
      jaccards)
check("airway_dex Jaccard", round(jaccards["airway_dex"], 5) == .99963,
      jaccards["airway_dex"])
check("GTEx Jaccard", round(jaccards["gtex_blood_muscle"], 5) == .99997,
      jaccards["gtex_blood_muscle"])

# Ranking and called-set claims used by the concordance figure.
top_overlaps = []
continuous_jaccards = []
for case, _ in labels:
    r = pd.read_csv(ROOT / f"bench/cache/{case}/r_results.csv", index_col="gene")
    o = pd.read_csv(
        ROOT / f"bench/cache/{case}/cu_reference_results.csv", index_col=0
    ).reindex(r.index)
    top = max(20, min(2000, len(r) // 4))
    ns = np.unique(np.round(np.logspace(1, np.log10(top), 14)).astype(int))
    r_order = np.argsort(-np.abs(r["stat"].to_numpy()), kind="stable")
    o_order = np.argsort(-np.abs(o["stat"].to_numpy()), kind="stable")
    for n in ns:
        top_overlaps.append(len(set(r_order[:n]) & set(o_order[:n])) / n)

    rp = r["padj"].to_numpy()
    op = o["padj"].to_numpy()
    candidates = np.unique(np.concatenate([
        rp[np.isfinite(rp) & (rp >= 1e-3) & (rp <= .2)],
        op[np.isfinite(op) & (op >= 1e-3) & (op <= .2)],
        np.array([1e-3, .2]),
    ]))
    for alpha in candidates:
        rs = np.isfinite(rp) & (rp < alpha)
        os = np.isfinite(op) & (op < alpha)
        union = (rs | os).sum()
        continuous_jaccards.append((rs & os).sum() / union if union else 1.0)
check("top-N minimum over plotted values", min(top_overlaps) >= .996)
check("called-set minimum over continuous alpha range",
      min(continuous_jaccards) >= .961)

# The plotted airway diagnostic counts are retained in its source CSV/JSON.
airway_details = pd.read_csv(ROOT / "validation/data/airway/r_disp_details.csv")
airway_validation = json.loads(
    (ROOT / "validation/results/airway.json").read_text()
)
check("airway dispersion outliers", int(airway_details["dispOutlier"].sum()) == 141)
check("airway significant genes", airway_validation["n_sig_r_0.05"] == 3993)

# Counterfactual Wald covariance experiment.
without_floor = MINMU["without_minmu_floor"]
with_floor = MINMU["with_minmu_floor"]
check("minmu R calls", MINMU["R_calls"] == 206)
check("minmu unfloored calls", without_floor["calls"] == 202)
check("minmu unfloored intersection", without_floor["intersection_with_R"] == 201)
check("minmu unfloored R-only", without_floor["R_only"] == 5)
check("minmu unfloored cuDESeq2-only", without_floor["cuDESeq2_only"] == 1)
check("minmu unfloored Jaccard",
      round(without_floor["jaccard_with_R"], 3) == .971)
check("minmu floored exact called set", with_floor["jaccard_with_R"] == 1.0)


# Eager--Triton real-data dispersion differences retained by the A100 run.
mode_dispersion = {
    case: next(row for row in rows if row["substep"] == "dispersion")
    for case, rows in json.loads((ROOT / "bench/results/parity.json").read_text()).items()
}
check("GTEx eager-Triton maximum absolute dispersion difference",
      round(mode_dispersion["gtex_blood_muscle"]["gpu_modes_maxdiff"], 4) == .2604)
check("small-design eager-Triton maximum absolute dispersion difference",
      max(row["gpu_modes_maxdiff"] for case, row in mode_dispersion.items()
          if case != "gtex_blood_muscle") <= 3.2e-7)
# Every fact marker must correspond to one of the artifact computations above.
# Parse balanced braces so values such as ``3{,}993`` are not truncated.
def fact_values(source: str) -> set[str]:
    values: set[str] = set()
    position = 0
    marker = r"\fact{"
    while True:
        start = source.find(marker, position)
        if start < 0:
            return values
        cursor = start + len(marker)
        depth = 1
        end = cursor
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        values.add(re.sub(r"\s+", " ", source[cursor:end - 1]))
        position = end


fact_text = {value for value in fact_values(TEX) if not value.startswith("Reproduce:")}
artifact_checked = {
    "8.5e-4", "2.9--13.4$\\times$", "0.44\\%", "$8.1\\times10^{-5}$",
    "0.99722", "141", "3{,}993", "$\\ge$99.6\\%", "$\\ge$0.961",
    "0.99963", "0.99997", "4\\,ms", "8.7$\\times$", "2.7$\\times$",
    "730", "83", "1586", "583", "0.2604", "$3.2 \\times 10^{-7}$",
}
check("every manuscript fact has an artifact-backed computation",
      fact_text == artifact_checked,
      {"missing_checks": sorted(fact_text - artifact_checked),
       "stale_checks": sorted(artifact_checked - fact_text)})

print(f"{checks} checks run")
if fails:
    print(f"{len(fails)} mismatch(es):")
    for failure in fails:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ALL CHECKED MANUSCRIPT NUMBERS MATCH THEIR SOURCE DATA")
