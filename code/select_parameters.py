# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:47:45 2023

@author: qianyuqing
"""
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials,partial,rand
import numpy as np
import cupy as cp
import pandas as pd
from find_hv_genes import find_hv_genes
from take_norm import take_norm
from fold_cv import ten_fold_cv

dev = cp.cuda.Device(0)
print(dev)
dev.use()

data_dir = r'F:\scRNA-seq\paper\scRNMF-main\datasets\deng.csv'
expr = pd.read_csv(data_dir,index_col=False,header=None,low_memory=False)
label_name = expr.values[0,:]
label = label_name.copy()
X = expr.values[1:,:]
X = X.T
X = X.astype(np.int64)
label_class = np.unique(label)
j = 0
for i in label_class:
    label[label==i] = j
    j = j + 1
K = len(np.unique(label))

#filt gene zero expression
print('filt gene express...')
X_mean = np.mean(X,0)
X = X[:,X_mean!=0]

ncell,ngene = X.shape[0],X.shape[1]
print('{} cells, {} genes \n proportion of 0s: {} '.format(ncell,ngene,np.mean(X==0)))
#注意细胞行和列

# obtain 2000  HVGs. 
print('obtain 2000 high variable genes')
highvar_genes = find_hv_genes(X,top=2000)
X = X[:,highvar_genes]

# norm
print('take log norm...')
X,_ = take_norm(X)

def hyperopt_Closs_NMF_scRNA(param):
    X_train = X.copy()
    rmse = ten_fold_cv(X_train,param,10)
    str_results = 'Hyper parameters: '+'lammda:'+str(param['lammda'])+' beta:'+str(param['beta'])+' k:'+str(param['k'])+' sita:'+str(param['sita'])+ ' alpha:'+str(param['alpha'])+' rmse: '+str(rmse)
    print(str_results)
    del X_train,str_results
    return {'loss':rmse,'status': STATUS_OK}

space_Closs = {'lammda':hp.choice('lammda', [10,1,0.1,0.01,0.001,0.0001,0.00001]),
                  'alpha':hp.choice('alpha', [10,1,0.1,0.01,0.001,0.0001,0.00001]),
                  'beta':hp.choice('beta', [10,1,0.1,0.01,0.001,0.0001,0.00001]),
                  'k':hp.choice('k', [2,10,20,30,40,50]),
                  'sita':hp.choice('sita', [1000,10,1,0.1,0.01,0.001,0.0001,0.00001])}

algo = rand.suggest
trials = Trials()
best = fmin(hyperopt_Closs_NMF_scRNA, space_Closs,algo=algo, max_evals=1000, trials=trials)
print(best)
