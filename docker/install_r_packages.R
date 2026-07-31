options(repos = c(CRAN = "https://cloud.r-project.org"))

if (getRversion() != "4.6.0") {
  stop("container must use R 4.6.0; found ", getRversion())
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
if (as.character(BiocManager::version()) != "3.23") {
  stop("container must use Bioconductor 3.23; found ", BiocManager::version())
}

packages <- c(
  "DESeq2",
  "apeglm",
  "BiocParallel",
  "SummarizedExperiment",
  "pasilla",
  "airway"
)
BiocManager::install(packages, ask = FALSE, update = FALSE)

expected <- c(
  DESeq2 = "1.52.0",
  apeglm = "1.34.0",
  BiocParallel = "1.46.0"
)
actual <- vapply(names(expected), function(pkg) {
  as.character(utils::packageVersion(pkg))
}, character(1))
if (!identical(unname(actual), unname(expected))) {
  stop(
    "unexpected Bioconductor package versions: ",
    paste(names(actual), actual, sep = "=", collapse = ", ")
  )
}
