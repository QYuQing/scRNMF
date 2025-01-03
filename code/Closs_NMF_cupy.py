# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:50:46 2022

@author: qyq
"""
import cupy as cp
import numpy as np
import scipy.io as scio
from scipy import linalg
from kernel import kernel_gussian,kernel_cosine,kernel_corr, kernel_MI, kernel_normalized
from Tool import get_train_label,get_pre,get_auc_aupr,wknkn
import math
import time
class Closs_NMF_cupy(object):
    def __init__(self , v , param):
        
        self.v = v #交互矩阵
        
        self.m = cp.size(v[:,0]) #m行n列
        self.n = cp.size(v[0,:])
        
        self.lammda = param['lammda']/2 #二范数超参
        self.alpha = param['alpha'] #lap drug
        self.beta = param['beta']/2 # lap side effect
        self.k = param['k']
        self.sita = param['sita']
        # filter for unobserved associations
        self.un = cp.where(v==0,1,0)
        # filter for clinical trials values
        self.ct = cp.where(v>0,1,0)
        
        self.e = 10**-16 


    def random_init_parameter(self):
        #初始化参数
        w = cp.random.uniform(0 , 1 , [self.m , self.k])*0.1
        h = cp.random.uniform(0 , 1 , [self.k , self.n])*0.1
        return w,h
    

    def update_m(self,w,h):
        pre = self.reconstruct_v(w,h)
        e_2 = (self.v-pre)**2
        m = cp.exp( -(1/(2*self.sita**2)) * e_2 )
        m = cp.maximum(m,self.ct)
        return m
   
    def update_w(self, m,w,h,k_d): 
        fenzi = cp.dot( m*self.v , h.T ) + self.alpha*cp.dot( k_d , w )
        fenmu = cp.dot( m*cp.dot(w,h) + self.lammda*self.un*cp.dot(w,h), h.T) + self.alpha*cp.dot(cp.dot(w,w.T),w) + self.e
        w = w * (fenzi/fenmu)

        #w = np.maximum(w,0)
        return w

    def update_h(self, m,w,h,k_s):
        fenzi = cp.dot( w.T , m*self.v) + self.beta*cp.dot( h , k_s )
        fenmu = cp.dot(w.T,m*cp.dot(w,h)+self.lammda*self.un*cp.dot(w,h)) + self.beta*cp.dot(cp.dot(h,h.T),h) + self.e
        h = h * (fenzi/fenmu)
        
        return h

    
    def cal_sparse(self,w):
        G = cp.sum(w**2,1)**-0.5
        temp = cp.identity(len(G))
        G = temp*G
        return G    
    
    def reconstruct_v(self,w,h):
        pre = cp.dot( w , h)
        return pre
    

    def train(self,w,h,k_s,k_d,flag=0):

        for i in range(2000):
            w0 = w
            h0 = h
            m = self.update_m(w,h)
            w = self.update_w(m,w,h,k_s)
            h = self.update_h(m,w,h,k_d)
            

            if self.break_iteration(w0,h0,w,h):
                break
            else:
                continue
        return m,w,h
    
    def break_iteration(self,w0,h0,w,h):
        dw = cp.max( cp.abs(w0-w) / ( 10**-8 + cp.max(cp.abs(w0)) ) )
        dh = cp.max( cp.abs(h0-h) / ( 10**-8 + cp.max(cp.abs(h0)) ) )
        delta = cp.max(cp.array([dw,dh]))    
        tolx = 0.001
        if delta <= tolx:
            return True
        else:
            return False
