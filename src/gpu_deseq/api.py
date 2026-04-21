from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from formulaic import model_matrix
from scipy.stats import chi2, norm

MIN_MU = 1e-8
MIN_ALPHA = 1e-8
MAX_IRLS_STEPS = 50
IRLS_TOL = 1e-6


def _to_tensor(counts: object) -> torch.Tensor:
    if isinstance(counts, torch.Tensor):
        tensor = counts.detach().clone()
    else:
        tensor = torch.as_tensor(np.asarray(counts))
    if tensor.ndim != 2:
        raise ValueError("counts must be a 2D matrix of genes x samples")
    if torch.any(tensor < 0):
        raise ValueError("counts must be non-negative")
    return tensor.to(dtype=torch.float64)


def _as_dataframe(coldata: object, sample_ids: Iterable[str]) -> pd.DataFrame:
    if isinstance(coldata, pd.DataFrame):
        frame = coldata.copy()
    else:
        frame = pd.DataFrame(coldata)
    frame = frame.reset_index(drop=True)
    if len(frame) != len(list(sample_ids)):
        raise ValueError("coldata rows must match the number of samples")
    return frame


def _resolve_device(backend: str) -> torch.device:
    if backend != "torch":
        raise ValueError(f"unsupported backend: {backend}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DeseqResult:
    test_type: str
    design_columns: list[str]
    coefficients: torch.Tensor
    covariance: torch.Tensor | None
    log_likelihood: torch.Tensor
    base_mean: torch.Tensor
    contrast_vector: torch.Tensor | None = None
    reduced_log_likelihood: torch.Tensor | None = None
    degrees_of_freedom: int | None = None
    gene_ids: list[str] | None = None


class DESeqDataset:
    def __init__(
        self,
        counts: object,
        coldata: object,
        design: str,
        gene_ids: Iterable[str] | None = None,
        sample_ids: Iterable[str] | None = None,
        backend: str = "torch",
    ) -> None:
        counts_tensor = _to_tensor(counts)
        n_genes, n_samples = counts_tensor.shape
        if sample_ids is None:
            sample_ids = [f"sample_{idx}" for idx in range(n_samples)]
        if gene_ids is None:
            gene_ids = [f"gene_{idx}" for idx in range(n_genes)]
        self.sample_ids = list(sample_ids)
        self.gene_ids = list(gene_ids)
        if len(self.sample_ids) != n_samples:
            raise ValueError("sample_ids length must match sample count")
        if len(self.gene_ids) != n_genes:
            raise ValueError("gene_ids length must match gene count")
        self.design = design
        self.coldata = _as_dataframe(coldata, self.sample_ids)
        self._design_cpu = model_matrix(design, self.coldata, output="pandas")
        self.design_columns = list(self._design_cpu.columns)
        self.backend = backend
        self.device = _resolve_device(backend)
        self.counts = counts_tensor.to(self.device)
        self.design_matrix = torch.as_tensor(
            self._design_cpu.to_numpy(dtype=np.float64),
            device=self.device,
            dtype=torch.float64,
        )
        self.size_factors: torch.Tensor | None = None
        self.normalized_counts: torch.Tensor | None = None
        self.dispersions_gene_wise: torch.Tensor | None = None
        self.dispersion_trend: torch.Tensor | None = None
        self.dispersions: torch.Tensor | None = None

    @property
    def n_genes(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[1])

    def to(self, device: str | torch.device) -> "DESeqDataset":
        target = torch.device(device)
        self.device = target
        self.counts = self.counts.to(target)
        self.design_matrix = self.design_matrix.to(target)
        if self.size_factors is not None:
            self.size_factors = self.size_factors.to(target)
        if self.normalized_counts is not None:
            self.normalized_counts = self.normalized_counts.to(target)
        if self.dispersions_gene_wise is not None:
            self.dispersions_gene_wise = self.dispersions_gene_wise.to(target)
        if self.dispersion_trend is not None:
            self.dispersion_trend = self.dispersion_trend.to(target)
        if self.dispersions is not None:
            self.dispersions = self.dispersions.to(target)
        return self


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    mask = np.isfinite(pvalues)
    adjusted = np.full_like(pvalues, np.nan, dtype=np.float64)
    if not np.any(mask):
        return adjusted
    pv = pvalues[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    scaled = ranked * n / np.arange(1, n + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    adjusted_subset = np.empty_like(scaled)
    adjusted_subset[order] = scaled
    adjusted[mask] = adjusted_subset
    return adjusted


def fit_size_factors(dataset: DESeqDataset, method: str = "median_ratio") -> DESeqDataset:
    if method != "median_ratio":
        raise ValueError("only median_ratio is supported")
    counts = dataset.counts
    positive_mask = torch.all(counts > 0, dim=1)
    if not torch.any(positive_mask):
        raise ValueError("median ratio size factor estimation requires genes with all counts > 0")
    positive_counts = counts[positive_mask]
    geom_means = torch.exp(torch.mean(torch.log(positive_counts), dim=1))
    ratios = positive_counts / geom_means[:, None]
    size_factors = torch.median(ratios, dim=0).values
    size_factors = size_factors / torch.exp(torch.mean(torch.log(size_factors)))
    dataset.size_factors = size_factors
    dataset.normalized_counts = dataset.counts / size_factors[None, :]
    return dataset


def fit_dispersions(dataset: DESeqDataset, fit_type: str = "parametric") -> DESeqDataset:
    if fit_type != "parametric":
        raise ValueError("only parametric dispersion fitting is supported")
    if dataset.normalized_counts is None:
        raise ValueError("size factors must be estimated before dispersions")
    normed = dataset.normalized_counts
    mean = torch.mean(normed, dim=1).clamp_min(MIN_MU)
    var = torch.var(normed, dim=1, correction=1)
    gene_wise = ((var - mean) / (mean * mean)).clamp_min(MIN_ALPHA)
    x = (1.0 / mean).detach().cpu().numpy()
    y = gene_wise.detach().cpu().numpy()
    design = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    trend = torch.as_tensor(
        coeffs[0] + coeffs[1] / mean.detach().cpu().numpy(),
        device=dataset.device,
        dtype=torch.float64,
    ).clamp_min(MIN_ALPHA)
    shrunk = torch.exp(0.5 * (torch.log(gene_wise) + torch.log(trend))).clamp_min(MIN_ALPHA)
    dataset.dispersions_gene_wise = gene_wise
    dataset.dispersion_trend = trend
    dataset.dispersions = shrunk
    return dataset


def _chunk_slices(total: int, chunk_size: int) -> list[slice]:
    return [slice(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def _batched_irls(
    counts: torch.Tensor,
    x: torch.Tensor,
    offset: torch.Tensor,
    dispersions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_genes = counts.shape[0]
    n_params = x.shape[1]
    xt = x.transpose(0, 1)
    pinv_x = torch.linalg.pinv(x)
    beta = (pinv_x @ torch.log((counts + 0.1).transpose(0, 1))).transpose(0, 1)
    beta[:, 0] = beta[:, 0] - torch.mean(offset)

    eye = torch.eye(n_params, dtype=x.dtype, device=x.device) * 1e-6
    for _ in range(MAX_IRLS_STEPS):
        eta = beta @ xt + offset[None, :]
        mu = torch.exp(eta).clamp_min(MIN_MU)
        weights = (mu / (1.0 + dispersions[:, None] * mu)).clamp_min(1e-8)
        z = eta + (counts - mu) / mu
        xw = x[None, :, :] * weights[:, :, None]
        xtwx = torch.einsum("gsp,sq->gpq", xw, x) + eye[None, :, :]
        xtwz = torch.einsum("gsp,gs->gp", xw, z - offset[None, :])
        new_beta = torch.linalg.solve(xtwx, xtwz.unsqueeze(-1)).squeeze(-1)
        max_delta = torch.max(torch.abs(new_beta - beta)).item()
        beta = new_beta
        if max_delta < IRLS_TOL:
            break

    eta = beta @ xt + offset[None, :]
    mu = torch.exp(eta).clamp_min(MIN_MU)
    weights = (mu / (1.0 + dispersions[:, None] * mu)).clamp_min(1e-8)
    xw = x[None, :, :] * weights[:, :, None]
    fisher = torch.einsum("gsp,sq->gpq", xw, x) + eye[None, :, :]
    covariance = torch.linalg.pinv(fisher)
    log_likelihood = _nb_log_likelihood(counts, mu, dispersions)
    return beta, covariance, log_likelihood


def _nb_log_likelihood(counts: torch.Tensor, mu: torch.Tensor, dispersions: torch.Tensor) -> torch.Tensor:
    alpha = dispersions[:, None].clamp_min(MIN_ALPHA)
    theta = 1.0 / alpha
    log_prob = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * torch.log(theta / (theta + mu))
        + counts * torch.log(mu / (theta + mu))
    )
    return torch.sum(log_prob, dim=1)


def _fit_glm(dataset: DESeqDataset, design_matrix: torch.Tensor | None = None) -> DeseqResult:
    if dataset.dispersions is None or dataset.size_factors is None:
        raise ValueError("size factors and dispersions must be estimated before GLM fitting")
    x = dataset.design_matrix if design_matrix is None else design_matrix.to(dataset.device)
    offset = torch.log(dataset.size_factors)
    chunk_size = min(1024, dataset.n_genes)
    betas: list[torch.Tensor] = []
    covariances: list[torch.Tensor] = []
    log_likelihoods: list[torch.Tensor] = []
    for gene_slice in _chunk_slices(dataset.n_genes, chunk_size):
        beta, covariance, log_likelihood = _batched_irls(
            dataset.counts[gene_slice],
            x,
            offset,
            dataset.dispersions[gene_slice],
        )
        betas.append(beta)
        covariances.append(covariance)
        log_likelihoods.append(log_likelihood)
    coefficients = torch.cat(betas, dim=0)
    covariance = torch.cat(covariances, dim=0)
    log_likelihood = torch.cat(log_likelihoods, dim=0)
    base_mean = torch.mean(dataset.normalized_counts, dim=1)
    design_columns = dataset.design_columns if design_matrix is None else [f"coef_{i}" for i in range(x.shape[1])]
    return DeseqResult(
        test_type="fit",
        design_columns=design_columns,
        coefficients=coefficients,
        covariance=covariance,
        log_likelihood=log_likelihood,
        base_mean=base_mean,
        gene_ids=dataset.gene_ids,
    )


def _contrast_vector(result: DeseqResult, contrast: str | Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(contrast, str):
        try:
            idx = result.design_columns.index(contrast)
        except ValueError as exc:
            raise ValueError(f"unknown coefficient: {contrast}") from exc
        vector = torch.zeros(len(result.design_columns), dtype=torch.float64, device=result.coefficients.device)
        vector[idx] = 1.0
        return vector
    if isinstance(contrast, torch.Tensor):
        vector = contrast.to(device=result.coefficients.device, dtype=torch.float64)
    else:
        vector = torch.as_tensor(list(contrast), dtype=torch.float64, device=result.coefficients.device)
    if vector.shape != (len(result.design_columns),):
        raise ValueError("contrast vector has the wrong shape")
    return vector


def wald_test(dataset: DESeqDataset, contrast: str | Iterable[float] | torch.Tensor) -> DeseqResult:
    fitted = _fit_glm(dataset)
    contrast_vector = _contrast_vector(fitted, contrast)
    return DeseqResult(
        test_type="wald",
        design_columns=fitted.design_columns,
        coefficients=fitted.coefficients,
        covariance=fitted.covariance,
        log_likelihood=fitted.log_likelihood,
        base_mean=fitted.base_mean,
        contrast_vector=contrast_vector,
        gene_ids=dataset.gene_ids,
    )


def lrt_test(dataset: DESeqDataset, reduced_design: str) -> DeseqResult:
    full = _fit_glm(dataset)
    reduced_frame = model_matrix(reduced_design, dataset.coldata, output="pandas")
    reduced_design_matrix = torch.as_tensor(
        reduced_frame.to_numpy(dtype=np.float64),
        device=dataset.device,
        dtype=torch.float64,
    )
    reduced = _fit_glm(dataset, reduced_design_matrix)
    if full.coefficients.shape[1] <= reduced.coefficients.shape[1]:
        raise ValueError("reduced design must have fewer columns than the full design")
    return DeseqResult(
        test_type="lrt",
        design_columns=full.design_columns,
        coefficients=full.coefficients,
        covariance=full.covariance,
        log_likelihood=full.log_likelihood,
        reduced_log_likelihood=reduced.log_likelihood,
        degrees_of_freedom=full.coefficients.shape[1] - reduced.coefficients.shape[1],
        base_mean=full.base_mean,
        gene_ids=dataset.gene_ids,
    )


def results(
    fit: DeseqResult,
    contrast: str | Iterable[float] | torch.Tensor | None = None,
    alpha: float = 0.1,
    p_adjust: str = "bh",
) -> pd.DataFrame:
    if p_adjust != "bh":
        raise ValueError("only Benjamini-Hochberg adjustment is supported")
    gene_ids = fit.gene_ids or [f"gene_{idx}" for idx in range(fit.coefficients.shape[0])]
    base_mean = fit.base_mean.detach().cpu().numpy()
    if fit.test_type == "lrt":
        assert fit.reduced_log_likelihood is not None
        assert fit.degrees_of_freedom is not None
        stat = 2.0 * (fit.log_likelihood - fit.reduced_log_likelihood).detach().cpu().numpy()
        pvalue = chi2.sf(stat, fit.degrees_of_freedom)
        padj = _bh_adjust(pvalue)
        frame = pd.DataFrame(
            {
                "baseMean": base_mean,
                "log2FoldChange": np.nan,
                "lfcSE": np.nan,
                "stat": stat,
                "pvalue": pvalue,
                "padj": padj,
                "significant": padj <= alpha,
            },
            index=gene_ids,
        )
        return frame

    resolved_contrast = fit.contrast_vector
    if contrast is not None:
        resolved_contrast = _contrast_vector(fit, contrast)
    if resolved_contrast is None:
        raise ValueError("a contrast is required for Wald results")

    covariance = fit.covariance
    assert covariance is not None
    effects = (fit.coefficients @ resolved_contrast).detach().cpu().numpy()
    variances = torch.einsum("p,gpq,q->g", resolved_contrast, covariance, resolved_contrast).clamp_min(1e-12)
    se = torch.sqrt(variances).detach().cpu().numpy()
    stat = effects / se
    pvalue = 2.0 * norm.sf(np.abs(stat))
    padj = _bh_adjust(pvalue)
    frame = pd.DataFrame(
        {
            "baseMean": base_mean,
            "log2FoldChange": effects / np.log(2.0),
            "lfcSE": se / np.log(2.0),
            "stat": stat,
            "pvalue": pvalue,
            "padj": padj,
            "significant": padj <= alpha,
        },
        index=gene_ids,
    )
    return frame
