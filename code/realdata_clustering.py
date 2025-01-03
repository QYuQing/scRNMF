# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:02:17 2022

@author: qyq
"""
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.cluster import KMeans
from Tool import purity_score,JaccardInd
from Closs_NMF import Closs_NMF
from kernel import kernel_cosine,kernel_corr
from find_hv_genes import find_hv_genes
from take_norm import take_norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# preprocessed data, no need to normalization
data_dir = r'F:\scRNA-seq\AutoClass-master\datasets\data_for_clustering\Buettner.csv'
expr = pd.read_csv(data_dir,index_col=0)
X = expr.values[:,:-1] 
label = expr.values[:,-1]
K = len(np.unique(label)) 
unique_class = np.unique(label)

# Data pre-processing
print('Data pre-processing...')
highvar_genes = find_hv_genes(X,top=2000)
X = X[:,highvar_genes]

# compute cell and gene similarity matrix
print('compute the similarity matrix...')
cell_cosine = kernel_cosine(X, mu=0.005, sigma=0.002)
cell_corr = kernel_corr(X, mu=0.005, sigma=0.002)
cell_cosine[cell_corr<0] = 0
gene_cosine = kernel_cosine(X.T, mu=0.005, sigma=0.002)
gene_corr = kernel_corr(X.T, mu=0.005, sigma=0.002)
gene_corr[gene_corr<0] = 0
cell_sim = (cell_corr + cell_cosine)/2
gene_sim = (gene_corr + gene_cosine)/2

# start imputing data by scRNMF
# parameters setting
print('start imputing data by scRNMF...')
param = {}
param['lammda']=0.00001
param['alpha']=0.0001
param['beta']=0.01
param['k'] = 2
param['sita'] = 0.001      
clo = Closs_NMF(X , param)
w,h = clo.nndsvd_init(flag=1) # initialize W and H by nndsvd
m,w,h = clo.train(w, h, cell_sim, gene_sim, flag=0) # an iteration method to optimize
imputed_counts = clo.reconstruct_v(w,h)

# clustering
highvar_genes = find_hv_genes(imputed_counts,top=200)
imputed_counts_hv = imputed_counts[:,highvar_genes]
kmeans = KMeans(n_clusters = K,copy_x=False).fit(imputed_counts_hv)
cluster_label = kmeans.labels_
print('NMI:',round(normalized_mutual_info_score(label, cluster_label),4))
print('ARI:',round(adjusted_rand_score(label,cluster_label),4))


