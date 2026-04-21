# gpu-deseq

`gpu-deseq` is a DESeq2-style differential expression library for Python with GPU-aware numerical kernels built on PyTorch.

Current scope:
- count matrix input plus sample metadata
- formula-based fixed-effect designs
- median-ratio size factor estimation
- gene-wise dispersion estimation plus parametric trend shrinkage
- negative-binomial GLM fitting
- Wald and likelihood-ratio tests
- tabular results with Benjamini-Hochberg correction

Example:

```python
from gpu_deseq import (
    DESeqDataset,
    fit_dispersions,
    fit_size_factors,
    lrt_test,
    results,
    wald_test,
)

dds = DESeqDataset(counts, coldata, design="~ batch + condition")
fit_size_factors(dds)
fit_dispersions(dds)
wald = wald_test(dds, contrast="condition[T.treated]")
res = results(wald)
```

The implementation is compatibility-focused rather than a line-by-line port of Bioconductor DESeq2.
