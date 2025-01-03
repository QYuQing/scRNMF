# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 22:09:11 2023

@author: qianyuqing
"""
import numpy as np
import random
from Closs_NMF import Closs_NMF
from Closs_NMF_cupy import Closs_NMF_cupy
import cupy as cp
from kernel import kernel_cosine,kernel_corr

def masking(X,param,rate_list):
    # X需要正则化
    # 行是细胞，列是基因
    # rate_list = [0.02,0.05,0.1]
    
    n_cell = len(X)
    n_gene = len(X[0])
    #五折交叉,随机遮掉交互矩阵y的元素为0
    posi_all = []
    for i in range(n_cell):
        for j in range(n_gene):
            if X[i,j]>0:
                posi_all.append([i,j])

    random.shuffle(posi_all)
    n_posi = len(posi_all)
    
    pcc_list = []
    
    for rate in rate_list:
        cv = 1/rate
        print('begin rate: ',rate)
        step = round(n_posi/cv)
        rmse_list = []
        
        i = 0
        posi_test = posi_all[i*step:(i+1)*step]
    
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
        m,w,h = clo_cupy.train(w, h, cell_sim, gene_sim, flag=0)  
        imputed_counts = clo_cupy.reconstruct_v(w,h)
        imputed_counts = np.asnumpy(imputed_counts)
        
        pre = []
        for posi_test_index in posi_test:
            pre.append(imputed_counts[posi_test_index[0],posi_test_index[1]])
        
        pcc =  np.corrcoef(label,pre)[0,1]
        print('Pearson correlation coefficient: '+str(pcc))
        pcc_list.append(pcc)
    
    return sum(pcc_list)/len(rate_list)