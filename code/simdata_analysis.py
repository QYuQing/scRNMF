# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:27:57 2023

@author: qianyuqing
"""
from Closs_NMF import Closs_NMF
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from kernel import kernel_cosine,kernel_corr
from sklearn.decomposition import PCA
import umap
from find_hv_genes import find_hv_genes
import seaborn as sns
from take_norm import take_norm

#load dataset
print('load data...')
sim_name = '6'
counts = pd.read_csv('F:\\scRNA-seq\\Dataset1\\sim.groups'+sim_name+'\\sim.groups'+sim_name+'_counts.csv')
counts = counts.iloc[:,1:]
counts = np.array(counts)

dropout_pro = pd.read_csv('F:\\scRNA-seq\\Dataset1\\sim.groups'+sim_name+'\\sim.groups'+sim_name+'_Dropout_Pro.csv')
dropout_pro = dropout_pro.iloc[:,1:]
dropout_pro = np.array(dropout_pro)
dropout_pro = dropout_pro.T

true_counts = pd.read_csv('F:\\scRNA-seq\\Dataset1\\sim.groups'+sim_name+'\\sim.groups'+sim_name+'_truecounts.csv')
true_counts = true_counts.iloc[:,1:]
true_counts = np.array(true_counts)
true_counts = true_counts.T

true_group_data = pd.read_csv('F:\\scRNA-seq\\Dataset1\\sim.groups'+sim_name+'\\sim.groups'+sim_name+'_groups.csv')
true_group_data = true_group_data.iloc[:,1:]
true_group_data = np.array(true_group_data)
true_group = []
for i in true_group_data:
    true_group.append(int(i[0][-1]))
K = len(np.unique(true_group))

# Data pre-processing
print('Data pre-processing...')
X,norm_factor = take_norm(counts.T,cellwise_norm=True, log1p=True)   

# compute cell and gene similarity matrix
print('compute the similarity matrix...')
cell_cosine = kernel_cosine(X, mu=0.005, sigma=0.002)
# cell_MI = kernel_MI(X)
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
param['beta']=0.001
param['k'] = 10
param['sita'] = 0.01      
clo = Closs_NMF(X , param)
w,h = clo.nndsvd_init(flag=1) # initialize W and H by nndsvd
m,w,h = clo.train(w, h, cell_sim, gene_sim, flag=0) # an iteration method to optimize
imputed_counts = clo.reconstruct_v(w,h)

# downstream analysis
print('start downstream analysis...')
# measure by RMSE and PCC
pre = imputed_counts[dropout_pro]
true_counts_norm = np.log(np.dot(norm_factor, true_counts)+1)
label = true_counts_norm[dropout_pro]
rmse = mean_squared_error(label, pre)**0.5
pcc = np.corrcoef(pre,label)
print('recovering RMSE:',str(rmse),str(pcc[0,1]))

# measure by visualize
df_imputed_counts = pd.DataFrame(imputed_counts)
highvar_genes = find_hv_genes(imputed_counts,top=200)
imputed_counts = imputed_counts[:,highvar_genes]
umap_reducer = umap.UMAP(random_state=2023)
PCA_counts = PCA(n_components=50).fit_transform(imputed_counts)
umap_counts = umap_reducer.fit_transform(PCA_counts)
df_counts = pd.DataFrame({'Dimension1':umap_counts[:,0],'Dimension2':umap_counts[:,1],'label':true_group})
sns.scatterplot(data=df_counts, x='Dimension1', y='Dimension2', hue='label',legend=False,palette=sns.color_palette(n_colors=4))
