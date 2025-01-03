# scRNMF
###scRNMF: an imputation method for single-cell RNA-seq data by robust and non-negative matrix factorization

### Overview
Single-cell RNA-sequencing (scRNA-seq) technology provides a powerful tool for investigating cell heterogeneity and cell
subpopulations by allowing the quantification of gene expression at single-cell level. However, scRNA-seq data analysis
remains challenging because of dropout events. To address this issue, much effort has been done and some imputation
methods were developed. However, there is no outstanding imputation method for improving the downstream analysis
at present. Therefore, we propose an imputation method based robust and non-negative matrix factorization (scRNMF).
Different the common matrix factorization algorithm, we use the correntroy induced loss (C-loss) to fit zero counts,
which is much more insensitive to zero values which caused by dropout events. We test scRNMF and compare it with
other state-of-the-art methods on simulated and real datasets of various sizes and zero rates. scRNMF achieves the best
performance in the following scRNA-Seq data downstream analysis: gene expression recovering, cell clustering, gene
differential expression, and cellular trajectory reconstruction. We demonstrate that scRNMF is a powerful and stable
scRNA-seq data imputation tool.

### Installation
You can download scRNMF module from figshare.
AutoClass runs with Python 3.8, and you need to have a few other pacakges installed first (requirements.txt).


### Usage
In this repository, you can find several tutorials on scRNMF with Full examples.
simdata_analysis.py [Gene expression data recovery]
realdata_clustering.py [Cell clustering analysis]
select_parameters.py [Parameters selection]

Thank you for your interest.

You can contact author by email usts_qyq@qq.com!

Thank you for your interest.
