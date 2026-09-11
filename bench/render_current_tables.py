"""Render tables using only committed benchmark and validation artifacts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench/results"
VALIDATION_RESULTS = ROOT / "validation/results"
MODES = ("eager", "graph", "triton")
STEPS = ("normalization", "dispersion", "glm_fit", "significance", "lfc_shrink")


def number(value: float) -> str:
    return f"{value:.0f}"


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
        "## Matched sum of stage medians (ms)",
        "",
        "R and cuDESeq2 use the same five stage boundaries. These totals are sums "
        "of independently measured stage medians, not direct end-to-end observations.",
        "",
        "| dataset | P | n | R, 1 worker | eager | graph | Triton | best cuDESeq2 vs. R |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for case in cases:
        meta = metadata[case]
        r_total = timings["r"][case].get(
            "stage_total", timings["r"][case]["total"]
        )
        gpu = {
            mode: timings["cu"][case][mode].get(
                "stage_total", timings["cu"][case][mode]["total"]
            )
            for mode in MODES
        }
        speedup = r_total / min(gpu.values())
        lines.append(
            f"| {case} | {meta.get('P', '—')} | {meta['n_samples']} | "
            f"{r_total:.0f} | "
            + " | ".join(number(gpu[mode]) for mode in MODES)
            + f" | {speedup:.1f}× |"
        )

    lines += [
        "",
        "## Direct end-to-end wall time (ms)",
        "",
        "Each cell is the median of five complete workflow observations, including "
        "dataset construction and host-to-device transfer. These values are kept "
        "separate from the matched sums of stage medians above.",
        "",
        "| dataset | n | R, 1 worker | eager | graph | Triton | best cuDESeq2 vs. R |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for case in cases:
        direct_gpu = {
            mode: timings["cu"][case][mode]["total"] for mode in MODES
        }
        direct_r = timings["r"][case]["total"]
        speedup = direct_r / min(direct_gpu.values())
        lines.append(
            f"| {case} | {metadata[case]['n_samples']} | {direct_r:.0f} | "
            + " | ".join(number(direct_gpu[mode]) for mode in MODES)
            + f" | {speedup:.1f}× |"
        )

    lines += [
        "",
        f"## Direct R end-to-end wall time: one vs. {workers} workers (ms)",
        "",
        "Each cell is the median of direct `DESeq() + results() + lfcShrink()` "
        "observations. This separately collected table is not combined with the "
        "stage-summed GPU measurements above.",
        "",
        f"| dataset | n | 1 worker | {workers} workers |",
        "|---|--:|--:|--:|",
    ]
    for case in cases:
        record = parallel["cases"][case]
        lines.append(
            f"| {case} | {metadata[case]['n_samples']} | "
            f"{record['serial']['median_ms']:.0f} | "
            f"{record['parallel']['median_ms']:.0f} |"
        )

    lines += [
        "",
        "## Per-stage wall time (ms)",
        "",
        "| dataset | substep | R | eager | graph | Triton |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for case in cases:
        for step in STEPS:
            values = [timings["cu"][case][mode][step] for mode in MODES]
            lines.append(
                f"| {case} | {step} | {timings['r'][case][step]:.0f} | "
                + " | ".join(number(value) for value in values)
                + " |"
            )

    lines += [
        "",
        "## Output parity against DESeq2 1.52.0",
        "",
        "The retained reference-parity artifact scores eager mode; graph and Triton "
        "differences are retained separately in `parity.json`.",
        "",
        "| dataset | substep | metric | eager vs. R | tolerance | verdict |",
        "|---|---|---|--:|--:|:--:|",
    ]
    for case in cases:
        for metric in parity["cases"][case]["metrics"]:
            comparison = "≥" if metric["higher_better"] else "≤"
            verdict = "PASS" if metric["pass"] else "FAIL"
            lines.append(
                f"| {case} | {metric['substep']} | {metric['metric']} | "
                f"{metric['value']:.6g} | {comparison}{metric['tol']:g} | "
                f"{verdict} |"
            )

    destination = RESULTS / "TABLES.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
