# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:02:50 2022

@author: qyq
"""
import pandas as pd
import numpy as np
from find_hv_genes import find_hv_genes
from fold_cv import ten_fold_cv_analysis
import math
from estimate_k import estimate_k
# preprocessed data, no need to normalization
data_dir = r'E:\scRNA-seq\AutoClass-master\datasets\data_for_clustering\Buettner.csv'
# data_dir = r'E:\scRNA-seq\AutoClass-master\datasets\data_for_clustering\Usoskin.csv'
# data_dir = r'E:\scRNA-seq\AutoClass-master\datasets\data_for_clustering\Lake.csv'
# data_dir = r'E:\scRNA-seq\AutoClass-master\datasets\data_for_clustering\Zeisel.csv'
print(data_dir)
expr = pd.read_csv(data_dir,index_col=0)
X = expr.values[:,:-1] 
label = expr.values[:,-1]
K = len(np.unique(label)) 
unique_class = np.unique(label)
#注意细胞行和列
ncell,ngene = X.shape[0],X.shape[1]
print('{} cells, {} genes \n proportion of 0s: {} \n {} different cell types'.format(ncell,ngene,np.mean(X==0),K))

# obtain 2000  HVGs. 
print('obtain 2000 high variable genes')
highvar_genes = find_hv_genes(X,top=2000)
X = X[:,highvar_genes]

k_1 = math.ceil(ncell/10)
k_2 = estimate_k(X)

param = {'lammda':0.00001,'beta':0.01,'k':2,'sita':0.001,'alpha':0.0001}
param_list={'lammda':[10,1,0.1,0.01,0.001,0.0001,0.00001],
            'beta':[10,1,0.1,0.01,0.001,0.0001,0.00001],
            'k':[k_1,k_2,2,10,20,30,40,50],
            'sita':[1000,10,1,0.1,0.01,0.001,0.0001,0.00001],
            'alpha':[10,1,0.1,0.01,0.001,0.0001,0.00001]}
param_str_list = ['lammda','beta','k','sita','alpha']
cv = 10
results_list = ten_fold_cv_analysis(X,param,param_list,param_str_list,cv)
np.save('Buettner_para_analysis.npy',results_list)





