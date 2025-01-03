# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:06:45 2022

@author: qyq
"""
import numpy as np
import random
from Closs_NMF import Closs_NMF
from Closs_NMF_cupy import Closs_NMF_cupy
import cupy as cp
from kernel import kernel_cosine,kernel_corr
from sklearn.metrics import mean_squared_error
def ten_fold_cv(X,param,cv): 
    n_cell = len(X)
    n_gene = len(X[0])
    posi_all = []
    for i in range(n_cell):
        for j in range(n_gene):
            if X[i,j]>0:
                posi_all.append([i,j])

    random.shuffle(posi_all)
    n_posi = len(posi_all)
    step = round(n_posi/cv)
    
    rmse_list = []
    for i in range(cv):
        print('begin cv: '+str(i))
        
        if i!=(cv-1):
            posi_test = posi_all[i*step:(i+1)*step]
        else:
            posi_test = posi_all[i*step:n_posi]
    
        X_train = np.copy(X)
        label = []
        for posi_test_index in posi_test:
            label.append(X_train[posi_test_index[0],posi_test_index[1]])
            X_train[posi_test_index[0],posi_test_index[1]] = 0
        
        # print('compute the similarity matrix...')
        cell_cosine = kernel_cosine(X_train, mu=0.005, sigma=0.002)
        cell_corr = kernel_corr(X_train, mu=0.005, sigma=0.002)
        cell_cosine[cell_corr<0] = 0
        gene_cosine = kernel_cosine(X_train.T, mu=0.005, sigma=0.002)
        gene_corr = kernel_corr(X_train.T, mu=0.005, sigma=0.002)
        gene_corr[gene_corr<0] = 0
        cell_sim = (cell_corr + cell_cosine)/2
        gene_sim = (gene_corr + gene_cosine)/2
        # print('init parameter...')
        clo = Closs_NMF(X_train , param)
        w,h = clo.nndsvd_init(flag=1)
        
        cell_sim,gene_sim,w,h,X_cupy = cp.asarray(cell_sim),cp.asarray(gene_sim),cp.asarray(w),cp.asarray(h),cp.asarray(X_train)
        
        # print('begin train')
        clo_cupy = Closs_NMF_cupy(X_cupy , param)
        m,w,h,loss_list,value_list = clo_cupy.train(w, h, cell_sim, gene_sim, flag=0)  
        imputed_counts = clo_cupy.reconstruct_v(w,h)
        imputed_counts = cp.asnumpy(imputed_counts)
        
        pre = []
        for posi_test_index in posi_test:
            pre.append(imputed_counts[posi_test_index[0],posi_test_index[1]])
        
        rmse = mean_squared_error(label,pre)**0.5
        print('RMSE: '+str(rmse))
        rmse_list.append(rmse)
    
    return sum(rmse_list)/cv
    
    
    
    
    
    
    
    
