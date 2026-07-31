"""Check manuscript numbers against the committed benchmark artifacts.

This audit is deliberately read-only.  It checks the current DESeq2 reference
version, the two manuscript tables derived from the real-data run, the headline
speedup and parity summaries, and the synthetic optimization claims.  It also
requires every ``\fact{}`` marker to have a named source class.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper/main.tex").read_text()
TEXN = re.sub(r"\s+", " ", TEX)
PARITY = json.loads((ROOT / "bench/results/reference_parity.json").read_text())
TIMING = json.loads((ROOT / "bench/results/timings.json").read_text())
ROOF = json.loads((ROOT / "benchmarks/results_roofline.json").read_text())
GENE = json.loads((ROOT / "benchmarks/results_a100.json").read_text())["results"]
SAMPLE = json.loads(
    (ROOT / "benchmarks/results_a100_samplesweep.json").read_text()
)["sample_sweep"]
CHUNK = json.loads(
    (ROOT / "benchmarks/results_a100_sweep.json").read_text()
)["sweep"]

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
for case, record in PARITY["cases"].items():
    check(f"{case}: DESeq2 version", record["deseq2_version"] == "1.52.0")
    check(f"{case}: R version", record["r_version"] == "4.6.0")
    check(f"{case}: apeglm version", record["apeglm_version"] == "1.34.0")
check("standard reference pipeline",
      PARITY["pipeline"] == "standard_wald_with_outlier_refit")


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
    check(f"Table 3 {case}: R seconds",
          round(values[2], 1) == round(TIMING["r"][case]["total"] / 1000, 1),
          values)
    shown_gpu = values[3:6]
    actual_gpu = [TIMING["cu"][case][mode]["total"]
                  for mode in ("eager", "graph", "triton")]
    check(f"Table 3 {case}: GPU totals",
          all(round(a) == round(b) for a, b in zip(shown_gpu, actual_gpu)),
          (shown_gpu, actual_gpu))
    speedups.append(TIMING["r"][case]["total"] / min(actual_gpu))

    block_start = substep_table.index(label + " &")
    next_mid = substep_table.find(r"\midrule", block_start)
    block = substep_table[block_start:next_mid if next_mid >= 0 else None]
    for stage in ("dispersion", "glm_fit", "significance", "lfc_shrink"):
        stage_label = stage.replace("_", r"\_")
        line = next(line for line in block.splitlines() if stage_label in line)
        shown = [int(x) for x in re.findall(r"&\s*(\d+)", line)]
        actual = [round(TIMING["r"][case][stage])] + [
            round(TIMING["cu"][case][mode][stage])
            for mode in ("eager", "graph", "triton")
        ]
        check(f"Table 4 {case}/{stage}", shown == actual, (shown, actual))

headline = re.findall(r"\\fact\{10\.3--99\.2\$\\times\$\}", TEX)
check("headline range appears three times", len(headline) == 3, len(headline))
check("headline minimum", round(min(speedups), 1) == 10.3, min(speedups))
check("headline maximum", round(max(speedups), 1) == 99.2, max(speedups))


# Called-set claims are derived from the current standard-pipeline CSVs.
jaccards = {}
for case, _ in labels:
    r_path = ROOT / f"bench/cache/{case}/r_results.csv"
    o_path = ROOT / f"bench/cache/{case}/cu_reference_results.csv"
    import pandas as pd
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


# Synthetic optimization claims retained from the existing A100 artifacts.
rf = {row["G"]: row for row in ROOF["cases"]}
check("roofline sustained bandwidth", round(ROOF["sustained_gbs"]) == 1369)
check("roofline percent of specification",
      round(ROOF["sustained_pct_of_spec"]) == 88)
for key, shown in (("elementwise_pct_sustained", 21),
                   ("lgamma_pct_sustained", 43),
                   ("digamma_pct_sustained", 36)):
    check(f"roofline 20k {key}", round(rf[20000][key]) == shown,
          rf[20000][key])
check("roofline special-function times",
      round(rf[20000]["lp_and_dlp_us"]) == 1287
      and round(rf[2000]["lp_and_dlp_us"]) == 1292)

sample = {row["n_samples"]: row for row in SAMPLE}
check("sample-axis graph endpoints",
      round(sample[4]["fit_alpha_mle_speedup"], 1) == 7.6
      and round(sample[2000]["fit_alpha_mle_speedup"], 2) == .97)
check("sample-axis Triton endpoints",
      round(sample[4]["fit_alpha_mle_triton_speedup"], 1) == 67.8
      and round(sample[2000]["fit_alpha_mle_triton_speedup"], 1) == 6.4)
check("sample-axis iteration cap",
      sample[4]["eager_iters"] == sample[6]["eager_iters"] == 100)
converged = [row["eager_iters"] for n, row in sample.items() if n >= 30]
check("sample-axis converged range", (min(converged), max(converged)) == (13, 25))

gene = {(row["n_genes"], row["P"]): row for row in GENE}
for p, lo_ms, hi_ms, growth in ((2, 128, 212, 1.7), (4, 195, 225, 1.2)):
    lo, hi = gene[(2000, p)], gene[(20000, p)]
    check(f"gene-axis P={p} times",
          round(lo["fit_dispersions_eager_ms"]) == lo_ms
          and round(hi["fit_dispersions_eager_ms"]) == hi_ms)
    check(f"gene-axis P={p} growth",
          round(hi["fit_dispersions_eager_ms"] /
                lo["fit_dispersions_eager_ms"], 1) == growth)

for row in CHUNK:
    for entry in row["sweep"]:
        expected = math.ceil(row["eager_iters"] / entry["chunk"]) * entry["chunk"]
        check(f"chunk padding {row['n_genes']}/{entry['chunk']}",
              entry["graph_iters"] == expected)
        check(f"chunk parity {row['n_genes']}/{entry['chunk']}",
              entry["max_abs_diff"] == 0)


# Every fact marker must belong to a checked or explicitly test-backed class.
fact_text = {
    re.sub(r"\s+", " ", value)
    for value in re.findall(r"\\fact\{([^}]*)\}", TEX)
}
allowed = {
    "$0.016$", "$0.080$", "$0.5285285285285285$", "$\\approx$0.3\\,s",
    "$\\ge$0.961", "$\\ge$99.6\\%", "0", "0.03\\%", "0.045\\%", "0.07",
    "0.44\\%", "0.97$\\times$", "0.971", "0.997", "0.99963", "0.99997",
    "1.000", "1.01$\\times$", "1.2$\\times$", "1.4\\%", "1.7$\\times$",
    "1.8$\\times$", "1.84$\\times$", "10.3--99.2$\\times$", "100", "128",
    "1287\\,\\textmu s", "1292\\,\\textmu s", "13--25", "1369\\,GB/s",
    "141", "195", "1e-14", "2.2$\\times$", "2.45$\\times$", "206", "212",
    "21\\%", "225", "22\\%", "25", "3--8\\%", "3.0e-5", "3.0e-8",
    "3.7$\\times$", "31", "36\\%", "3e-14", "3e-15", "3{,", "4.4e-3",
    "494--791\\,ms", "4.5$\\times$", "4.5e-4", "43\\%", "470\\,ms",
    "5.57$\\times$", "5e-15", "6.4$\\times$", "6.7\\%", "67.8$\\times$",
    "7.6$\\times$", "76/76 step-parity tests", "8.5e-4", "88\\%",
}
unregistered = sorted(
    value for value in fact_text
    if not value.startswith("Reproduce:") and value not in allowed
)
check("all fact markers registered", not unregistered, unregistered)

print(f"{checks} checks run")
if fails:
    print(f"{len(fails)} mismatch(es):")
    for failure in fails:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ALL CHECKED MANUSCRIPT NUMBERS MATCH THEIR SOURCE DATA")
