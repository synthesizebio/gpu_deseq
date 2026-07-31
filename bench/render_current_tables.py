"""Render the current standard-pipeline benchmark and parity tables."""
from __future__ import annotations

import json
from pathlib import Path


RESULTS = Path("bench/results")
DATA = Path("validation/data")
MODES = ("eager", "graph", "triton")
STEPS = ("normalization", "dispersion", "glm_fit", "significance", "lfc_shrink")


def number(value: float) -> str:
    return f"{value:.0f}"


def main() -> None:
    timings = json.loads((RESULTS / "timings.json").read_text())
    parity = json.loads((RESULTS / "reference_parity.json").read_text())
    cases = sorted(parity["cases"])
    lines = [
        "# Current standard-pipeline results",
        "",
        "R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. "
        "GPU timings are from one A100-SXM4-40GB standard-pipeline run.",
        "",
        "## End-to-end wall time (ms)",
        "",
        "| dataset | P | n | R | eager | graph | Triton | best measured speedup |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for case in cases:
        meta = json.loads((DATA / case / "meta.json").read_text())
        r_total = timings["r"][case]["total"]
        gpu = {mode: timings["cu"][case][mode]["total"] for mode in MODES}
        speedup = f"{r_total / min(gpu.values()):.1f}×"
        lines.append(
            f"| {case} | {meta.get('P', '—')} | {meta['n_samples']} | "
            f"{r_total:.0f} | "
            + " | ".join(number(gpu[mode]) for mode in MODES)
            + f" | {speedup} |"
        )

    lines += [
        "",
        "## Per-substep wall time (ms)",
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
        "| dataset | substep | metric | value | tolerance | verdict |",
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
