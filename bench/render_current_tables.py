"""Render tables using only committed benchmark and validation artifacts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench/results"
VALIDATION_RESULTS = ROOT / "validation/results"
MODES = ("eager", "graph", "triton")
STEPS = ("normalization", "dispersion", "glm_fit", "significance", "lfc_shrink")
STEP_LABELS = {
    "normalization": "normalization",
    "dispersion": "dispersion",
    "glm_fit": "GLM fit",
    "significance": "significance",
    "lfc_shrink": "LFC shrinkage",
}


def number(value: float) -> str:
    return f"{value:.0f}"


def display_step(step: str) -> str:
    return STEP_LABELS.get(step, step.replace("_", " "))


def main() -> None:
    timings = json.loads((RESULTS / "timings.json").read_text())
    parity = json.loads((RESULTS / "reference_parity.json").read_text())
    parallel = json.loads(
        (RESULTS / "r_parallel_a100_12worker.json").read_text()
    )
    cases = sorted(parity["cases"])
    metadata = {
        case: json.loads((VALIDATION_RESULTS / f"{case}.json").read_text())["meta"]
        for case in cases
    }
    workers = parallel["provenance"]["workers"]
    lines = [
        "# Current standard-pipeline results",
        "",
        "R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. GPU "
        "measurements use one A100-SXM4-40GB.",
        "",
        f"## Practical GPU acceleration versus {workers}-worker R (ms)",
        "",
        "Each cell is the median of five direct, complete workflow observations. "
        "The CPU baseline uses `BiocParallel::MulticoreParam(12)` with BLAS and "
        "OpenMP pinned to one thread per worker.",
        "",
        f"| dataset | P | n | R, {workers} workers | eager | graph | Triton | best GPU vs. R |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for case in cases:
        meta = metadata[case]
        r_total = parallel["cases"][case]["parallel"]["median_ms"]
        gpu = {
            mode: timings["cu"][case][mode]["total"]
            for mode in MODES
        }
        best_mode = min(gpu, key=gpu.get)
        speedup = r_total / min(gpu.values())
        lines.append(
            f"| {case} | {meta.get('P', '—')} | {meta['n_samples']} | "
            f"{r_total:.0f} | "
            + " | ".join(
                f"**{number(gpu[mode])}**" if mode == best_mode
                else number(gpu[mode])
                for mode in MODES
            )
            + f" | {speedup:.1f}× |"
        )

    lines += [
        "",
        "## GPU pipeline-stage measurements (ms)",
        "",
        "Bold values identify the fastest GPU mode for each measured stage. "
        "Totals sum independently measured stage medians and are not "
        "direct end-to-end observations.",
        "",
        "| dataset | stage | eager | graph | Triton |",
        "|---|---|--:|--:|--:|",
    ]
    for case in cases:
        for step in STEPS:
            gpu = {mode: timings["cu"][case][mode][step] for mode in MODES}
            best_mode = min(gpu, key=gpu.get)
            lines.append(
                f"| {case} | {display_step(step)} | "
                + " | ".join(
                    f"**{number(gpu[mode])}**" if mode == best_mode
                    else number(gpu[mode])
                    for mode in MODES
                )
                + " |"
            )
        totals = {
            mode: timings["cu"][case][mode].get(
                "stage_total", timings["cu"][case][mode]["total"]
            )
            for mode in MODES
        }
        best_mode = min(totals, key=totals.get)
        lines.append(
            f"| {case} | *total* | "
            + " | ".join(
                f"**{number(totals[mode])}**" if mode == best_mode
                else number(totals[mode])
                for mode in MODES
            )
            + " |"
        )

    lines += [
        "",
        "## Output parity against DESeq2 1.52.0",
        "",
        "The retained reference-parity artifact scores eager mode; graph and Triton "
        "differences are retained separately in `parity.json`.",
        "",
        "| dataset | stage | metric | eager vs. R | tolerance | verdict |",
        "|---|---|---|--:|--:|:--:|",
    ]
    for case in cases:
        for metric in parity["cases"][case]["metrics"]:
            comparison = "≥" if metric["higher_better"] else "≤"
            verdict = "PASS" if metric["pass"] else "FAIL"
            lines.append(
                f"| {case} | {display_step(metric['substep'])} | {metric['metric']} | "
                f"{metric['value']:.6g} | {comparison}{metric['tol']:g} | "
                f"{verdict} |"
            )

    destination = RESULTS / "TABLES.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
