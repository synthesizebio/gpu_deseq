"""Fail fast if a container no longer matches the paper's reference stack."""

from importlib.metadata import version
import subprocess

EXPECTED = {
    "formulaic": "1.2.2",
    "numba": "0.61.2",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "statsmodels": "0.14.6",
    "torch": "2.6.0+cu124",
    "triton": "3.2.0",
}

actual = {package: version(package) for package in EXPECTED}
wrong = {
    package: (EXPECTED[package], found)
    for package, found in actual.items()
    if found != EXPECTED[package]
}
if wrong:
    lines = [
        f"{package}: expected {expected}, found {found}"
        for package, (expected, found) in wrong.items()
    ]
    raise SystemExit("Python environment mismatch:\n  " + "\n  ".join(lines))

r_versions = subprocess.run(
    [
        "Rscript",
        "-e",
        (
            'cat(as.character(getRversion()), '
            'as.character(packageVersion("DESeq2")), '
            'as.character(packageVersion("apeglm")), '
            'as.character(packageVersion("BiocParallel")), '
            'as.character(packageVersion("SummarizedExperiment")), sep="\\n")'
        ),
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
expected_r = ["4.6.0", "1.52.0", "1.34.0", "1.46.0", "1.42.0"]
if r_versions != expected_r:
    raise SystemExit(
        f"R environment mismatch: expected {expected_r}, found {r_versions}"
    )

import torch

print("environment OK")
print(f"Python packages: {actual}")
print(
    f"R/DESeq2/apeglm/BiocParallel/SummarizedExperiment: {'/'.join(r_versions)}\n"
    f"CUDA build/runtime available: {torch.version.cuda}/{torch.cuda.is_available()}"
)
