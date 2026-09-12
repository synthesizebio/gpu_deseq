"""Fail fast if a container no longer matches the paper's reference stack."""

from importlib.metadata import version
import subprocess

EXPECTED = {
    "formulaic": "1.2.2",
    "numba": "0.64.0",
    "numpy": "2.4.6",
    "pandas": "2.3.3",
    "scipy": "1.17.1",
    "statsmodels": "0.14.6",
    "torch": "2.7.1",
    "triton": "3.3.1",
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

expected_torch_build = "2.7.1+cu126"
if torch.__version__ != expected_torch_build or torch.version.cuda != "12.6":
    raise SystemExit(
        "PyTorch CUDA build mismatch: expected "
        f"{expected_torch_build}/CUDA 12.6, found "
        f"{torch.__version__}/CUDA {torch.version.cuda}"
    )

print("environment OK")
print(f"Python packages: {actual}")
print(
    f"R/DESeq2/apeglm/BiocParallel/SummarizedExperiment: {'/'.join(r_versions)}\n"
    f"CUDA build/runtime available: {torch.version.cuda}/{torch.cuda.is_available()}"
)
