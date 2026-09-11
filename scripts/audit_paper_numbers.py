"""Check manuscript claims using only committed artifacts.

The audit deliberately does not read ignored prepared data or benchmark caches,
so it can run immediately after a clean checkout.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics


ROOT = Path(
    os.environ.get("GPU_DESEQ_AUDIT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
TEX = (ROOT / "paper/main.tex").read_text()
TEX_NORMALIZED = re.sub(r"\s+", " ", TEX)
RESULTS = ROOT / "bench/results"
TIMING = json.loads((RESULTS / "timings.json").read_text())
PARITY = json.loads((RESULTS / "reference_parity.json").read_text())
MODE_PARITY = json.loads((RESULTS / "parity.json").read_text())
PARALLEL_R = json.loads(
    (RESULTS / "r_parallel_a100_12worker.json").read_text()
)
SOURCE_MANIFEST_PATH = ROOT / "validation/data_sources.json"
SOURCE_MANIFEST = json.loads(SOURCE_MANIFEST_PATH.read_text())
PREPARED_MANIFEST = json.loads(
    (ROOT / "validation/prepared_data_manifest.json").read_text()
)
SAMPLE_SWEEP = json.loads(
    (ROOT / "benchmarks/results_a100_samplesweep_2026-09-11.json").read_text()
)

CASES = (
    ("pasilla", "pasilla"),
    ("pasilla_2fac", r"pasilla\_2fac"),
    ("airway_dex", r"airway\_dex"),
    ("airway_cell", r"airway\_cell"),
    ("airway", "airway"),
    ("gtex_blood_muscle", "gtex"),
)
MODES = ("eager", "graph", "triton")
STEPS = ("normalization", "dispersion", "glm_fit", "significance", "lfc_shrink")

failures: list[str] = []
checks = 0


def check(label: str, condition: bool, detail: object = "") -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(f"{label}: {detail}")


def close(a: float, b: float, rtol: float = 0.015) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-15)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def input_reconstruction(artifact: dict[str, object]) -> dict[str, object]:
    """Read the current schema plus the two historical manifest layouts."""
    provenance = artifact.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    reconstruction = provenance.get("input_reconstruction")
    if not isinstance(reconstruction, dict):
        reconstruction = artifact.get("input_reconstruction")
    if isinstance(reconstruction, dict):
        return reconstruction

    # Transitional writers used either flat provenance fields or
    # ``input_manifests: {source, prepared}``.
    source = provenance.get("source_data_manifest_sha256")
    prepared = provenance.get("prepared_data_manifest_sha256")
    manifests = provenance.get("input_manifests", artifact.get("input_manifests"))
    if isinstance(manifests, dict):
        source = source or manifests.get("source")
        prepared = prepared or manifests.get("prepared")
    return {
        "source_manifest_sha256": source,
        "prepared_manifest_sha256": prepared,
    }


def stage_total(record: dict[str, object]) -> float:
    """Prefer the revised explicit stage sum; legacy artifacts stored it as total."""
    return float(record.get("stage_total", record["total"]))


def table(label: str) -> str:
    match = re.search(
        rf"\\label\{{{re.escape(label)}\}}.*?\\end\{{tabular\}}", TEX, re.S
    )
    if not match:
        raise RuntimeError(f"table {label!r} not found")
    return match.group()


def numeric_cell(cell: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", cell, re.I)
    if not match:
        raise ValueError(f"no number in table cell {cell!r}")
    return float(match.group())


# Immutable input reconstruction contract.
check("source manifest schema", SOURCE_MANIFEST["schema_version"] == 1)
check("three public source records", len(SOURCE_MANIFEST["sources"]) == 3)
for source in SOURCE_MANIFEST["sources"]:
    check(f"{source['id']}: HTTPS source", source["url"].startswith("https://"))
    check(f"{source['id']}: positive byte length", source["size_bytes"] > 0)
    check(
        f"{source['id']}: SHA-256 shape",
        bool(re.fullmatch(r"[0-9a-f]{64}", source["sha256"])),
    )
check(
    "prepared manifest names source contract",
    PREPARED_MANIFEST["source_manifest"] == "validation/data_sources.json",
)
check(
    "prepared manifest pins source contract content",
    PREPARED_MANIFEST["source_manifest_sha256"] == digest(SOURCE_MANIFEST_PATH),
)
check(
    "prepared manifest covers six designs",
    set(PREPARED_MANIFEST["cases"]) == {case for case, _ in CASES},
)
for case, case_record in PREPARED_MANIFEST["cases"].items():
    files = case_record["files"]
    check(
        f"{case}: prepared file set",
        set(files) == {"counts.csv", "coldata.csv", "meta.json"},
    )
    for filename, record in files.items():
        check(f"{case}/{filename}: nonempty", record["size_bytes"] > 0)
        check(
            f"{case}/{filename}: SHA-256 shape",
            bool(re.fullmatch(r"[0-9a-f]{64}", record["sha256"])),
        )
gtex_metadata = PREPARED_MANIFEST["cases"]["gtex_blood_muscle"]["metadata"]
check("prepared GTEx seed", gtex_metadata["sample_seed"] == 1)
check(
    "prepared GTEx records 300 source columns",
    len(gtex_metadata["selected_source_columns"]) == 300,
)
check(
    "prepared GTEx records 300 source sample names",
    len(gtex_metadata["selected_source_samples"]) == 300,
)
check("paper names source manifest", "validation/data\\_sources.json" in TEX)
check("paper names prepared manifest", "validation/prepared\\_data\\_manifest.json" in TEX)
check("paper records GTEx seed", "using R seed 1" in TEX_NORMALIZED)
prepared_manifest_digest = digest(ROOT / "validation/prepared_data_manifest.json")
source_manifest_digest = digest(SOURCE_MANIFEST_PATH)
for artifact_name, artifact in (
    ("timings", TIMING),
    ("parallel R", PARALLEL_R),
    ("reference parity", PARITY),
):
    reconstruction = input_reconstruction(artifact)
    check(
        f"{artifact_name}: source manifest link",
        reconstruction.get("source_manifest_sha256") == source_manifest_digest,
    )
    check(
        f"{artifact_name}: prepared manifest link",
        reconstruction.get("prepared_manifest_sha256") == prepared_manifest_digest,
    )


# Reference provenance and parity thresholds. Retained legacy timing artifacts
# stored the stage sum in ``total``; newly generated artifacts store a direct
# observation in ``total`` and the matched sum in ``stage_total``.
timing_provenance = TIMING["provenance"]
legacy_stage_total = timing_provenance.get("direct_end_to_end") is False
if legacy_stage_total:
    check("legacy timing is labelled non-direct", True)
    check(
        "legacy GPU total definition",
        "sum of independently measured stage medians"
        in timing_provenance["cu_total_definition"],
    )
else:
    check(
        "current GPU total is direct",
        "median direct wall time" in timing_provenance.get("cu_total_definition", ""),
    )
    check(
        "current GPU stage-total definition",
        "sum of independently measured stage medians"
        in timing_provenance.get("cu_stage_total_definition", ""),
    )
    for case, record in TIMING["r"].items():
        check(f"{case}: current R stage total present", "stage_total" in record)
        check(f"{case}: current R direct samples present", bool(record.get("total_values_ms")))
    for case, modes in TIMING["cu"].items():
        for mode in MODES:
            record = modes[mode]
            check(f"{case}/{mode}: current GPU stage total present", "stage_total" in record)
            check(f"{case}/{mode}: current GPU direct samples present", bool(record.get("total_values_ms")))
            check(
                f"{case}/{mode}: current GPU stage total is stage sum",
                close(stage_total(record), sum(float(record[step]) for step in STEPS), 1e-9),
            )
check(
    "direct R worker total definition",
    "median direct wall time"
    in PARALLEL_R["provenance"]["total_definition"],
)
for case, record in TIMING["r"].items():
    check(
        f"{case}: staged R pipeline",
        record["pipeline"] == "staged_standard_wald_no_duplicate_dispersion",
    )
    check(f"{case}: staged output equivalence", record["output_equivalent_to_DESeq"] is True)
for case, record in PARITY["cases"].items():
    check(f"{case}: DESeq2 version", record["deseq2_version"] == "1.52.0")
    check(f"{case}: R version", record["r_version"] == "4.6.0")
    check(f"{case}: apeglm version", record["apeglm_version"] == "1.34.0")
    for metric in record["metrics"]:
        check(f"{case}/{metric['substep']}: parity pass", metric["pass"] is True)

metrics: dict[str, list[float]] = {}
directions: dict[str, bool] = {}
for record in PARITY["cases"].values():
    for row in record["metrics"]:
        metrics.setdefault(row["substep"], []).append(row["value"])
        directions[row["substep"]] = row["higher_better"]
check("30 real-data parity checks", sum(map(len, metrics.values())) == 30)

parity_table = table("tab:parity")
for step, values in metrics.items():
    row = next(
        line
        for line in parity_table.splitlines()
        if line.startswith(step.replace("_", r"\_") + " &")
    )
    cells = [cell.strip() for cell in row.split("&")]
    shown_median = numeric_cell(cells[-2])
    shown_worst = numeric_cell(cells[-1])
    actual_worst = min(values) if directions[step] else max(values)
    check(
        f"parity table {step}: median",
        close(shown_median, statistics.median(values), 0.06),
    )
    check(f"parity table {step}: worst", close(shown_worst, actual_worst, 0.06))


# Direct R-only table.
direct_table = table("tab:realtiming")
for case, label in CASES:
    row = next(
        line for line in direct_table.splitlines() if line.strip().startswith(label + " ")
    )
    cells = [cell.strip() for cell in row.split("&")]
    record = PARALLEL_R["cases"][case]
    check(
        f"direct R table {case}: one worker",
        round(numeric_cell(cells[3]), 1) == round(record["serial"]["median_ms"] / 1000, 1),
    )
    check(
        f"direct R table {case}: 12 workers",
        round(numeric_cell(cells[4]), 1) == round(record["parallel"]["median_ms"] / 1000, 1),
    )
    check(f"direct R table {case}: no GPU cells", len(cells) == 5)


# Matched per-stage table and stage-summed speed claim.
stage_table = table("tab:realsubstep")
speedups = []
for case, label in CASES:
    start = stage_table.index(label + " &")
    end = stage_table.find(r"\midrule", start)
    block = stage_table[start : end if end >= 0 else None]
    for step in STEPS[1:]:
        shown_row = next(
            line for line in block.splitlines() if step.replace("_", r"\_") in line
        )
        shown = [numeric_cell(cell) for cell in shown_row.split("&")[2:]]
        actual = [TIMING["r"][case][step]] + [
            TIMING["cu"][case][mode][step] for mode in MODES
        ]
        check(
            f"stage table {case}/{step}",
            all(round(a) == round(b) for a, b in zip(shown, actual)),
        )
    sum_row = next(line for line in block.splitlines() if "stage sum" in line)
    shown_sum = [numeric_cell(cell) for cell in sum_row.split("&")[2:]]
    actual_sum = [stage_total(TIMING["r"][case])] + [
        stage_total(TIMING["cu"][case][mode]) for mode in MODES
    ]
    check(
        f"stage table {case}/sum",
        all(round(a) == round(b) for a, b in zip(shown_sum, actual_sum)),
    )
    speedups.append(actual_sum[0] / min(actual_sum[1:]))
check("stage speed minimum", round(min(speedups), 1) == 8.4, min(speedups))
check("stage speed maximum", round(max(speedups), 1) == 58.9, max(speedups))
check(
    "headline stage range appears three times",
    TEX.count(r"\fact{8.4--58.9$\times$}") == 3,
)


# Sample-axis table has a clean, versioned primary artifact.
sample_provenance = SAMPLE_SWEEP["provenance"]
check("sample sweep clean checkout", sample_provenance["working_tree_dirty"] is False)
check("sample sweep seven repetitions", sample_provenance["n_timed"] == 7)
check("sample sweep three warmups", sample_provenance["n_warmup"] == 3)
check("sample sweep fixed at 2000 genes", all(row["n_genes"] == 2000 for row in SAMPLE_SWEEP["sample_sweep"]))
sample_table = table("tab:samplescaling")
for record in SAMPLE_SWEEP["sample_sweep"]:
    row = next(
        line
        for line in sample_table.splitlines()
        if line.strip().startswith(f"{record['n_samples']} ")
    )
    shown = [numeric_cell(cell) for cell in row.split("&")]
    actual = [
        record["n_samples"],
        record["fit_dispersions_eager_ms"],
        record["fit_dispersions_graph_ms"],
        record["fit_dispersions_triton_ms"],
    ]
    check(
        f"sample table n={record['n_samples']}",
        all(round(a) == round(b) for a, b in zip(shown, actual)),
    )


# Remaining quantitative claims.
validation = {
    case: json.loads((ROOT / f"validation/results/{case}.json").read_text())
    for case, _ in CASES
}
check("airway significant genes", validation["airway"]["n_sig_r_0.05"] == 3993)
check(
    "four exact called sets",
    sum(record["sig_jaccard_0.05"] == 1 for record in validation.values()) == 4,
)
check(
    "airway/dex Jaccard",
    round(validation["airway_dex"]["sig_jaccard_0.05"], 5) == 0.99963,
)
check(
    "GTEx Jaccard",
    round(validation["gtex_blood_muscle"]["sig_jaccard_0.05"], 5) == 0.99997,
)
check(
    "normalization at most 4 ms",
    max(
        TIMING["cu"][case][mode]["normalization"]
        for case, _ in CASES
        for mode in MODES
    )
    <= 4,
)
check(
    "reported Triton dispersion endpoints",
    round(TIMING["cu"]["pasilla_2fac"]["eager"]["dispersion"] / TIMING["cu"]["pasilla_2fac"]["triton"]["dispersion"], 1) == 8.7
    and round(TIMING["cu"]["airway"]["eager"]["dispersion"] / TIMING["cu"]["airway"]["triton"]["dispersion"], 1) == 2.7,
)
mode_dispersion = {
    case: next(row for row in rows if row["substep"] == "dispersion")
    for case, rows in MODE_PARITY.items()
}
check(
    "graph is bit-identical to eager across retained captures",
    all(
        row["maxdiff_vs_eager"]["graph"] == 0
        for rows in MODE_PARITY.values()
        for row in rows
    ),
)
check(
    "GTEx eager-Triton max dispersion difference",
    round(mode_dispersion["gtex_blood_muscle"]["gpu_modes_maxdiff"], 4) == 0.2604,
)
check(
    "smaller eager-Triton max dispersion difference",
    max(
        row["gpu_modes_maxdiff"]
        for case, row in mode_dispersion.items()
        if case != "gtex_blood_muscle"
    )
    <= 3.2e-7,
)


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
        values.add(re.sub(r"\s+", " ", source[cursor : end - 1]))
        position = end


facts = {value for value in fact_values(TEX) if not value.startswith("Reproduce:")}
checked_facts = {
    "8.5e-4",
    "0.99963",
    "0.99722",
    "8.4--58.9$\\times$",
    "0.44\\%",
    "$8.1\\times10^{-5}$",
    "3{,}993",
    "0.99997",
    "8.7$\\times$",
    "2.7$\\times$",
    "730",
    "83",
    "1586",
    "583",
    "4\\,ms",
    "0.2604",
    "$3.2 \\times 10^{-7}$",
}
check(
    "every marked fact is artifact-backed",
    facts == checked_facts,
    {"unchecked": sorted(facts - checked_facts), "stale": sorted(checked_facts - facts)},
)

print(f"{checks} checks run")
if failures:
    print(f"{len(failures)} mismatch(es):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ALL CHECKED MANUSCRIPT NUMBERS MATCH COMMITTED SOURCE ARTIFACTS")
