# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:57:26 2022

@author: qyq
"""
import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from nndsvd import nndsvd_init_parameter
class Closs_NMF(object):
    def __init__(self , v , param):
        
        self.v = v #交互矩阵
        
        self.m = np.size(v[:,0]) #m行n列
        self.n = np.size(v[0,:])
        
        self.lammda = param['lammda']/2 #二范数超参
        self.alpha = param['alpha'] #lap drug
        self.beta = param['beta']/2 # lap side effect
        self.k = param['k']
        self.sita = param['sita']
        # filter for unobserved associations
        self.un = np.int64(v==0)
        # filter for clinical trials values
        self.ct = np.int64(v>0)
        
        self.e = 10**-16 


    def random_init_parameter(self):
        #初始化参数
        print('random init')
        w = np.random.uniform(0 , 1 , [self.m , self.k])*0.1
        h = np.random.uniform(0 , 1 , [self.k , self.n])*0.1
        return w,h
    def nndsvd_init(self,flag):
        if flag==1:
            print('NNDSVDa: fill in the zero elements with the average')
        elif flag==2:
            print('NNDSVDar: fill in the zero elements with random values in the space [0:average/100]')
        else:
            print('NNDSVD: no fill')
        w,h = nndsvd_init_parameter(self.v,self.k,flag)
        return w,h
    def update_m(self,w,h):
        pre = self.reconstruct_v(w,h)
        v1 = self.v
        e_2 = (v1-pre)**2
        m = np.exp( -(1/(2*self.sita**2)) * e_2 )
        m = np.maximum(m,self.ct)
        return m
   
    def update_w(self, m,w,h,k_d): 
        fenzi = np.dot( m*self.v , h.T ) + self.alpha*np.dot( k_d , w )
        fenmu = np.dot( m*np.dot(w,h) + self.lammda*self.un*np.dot(w,h), h.T) + self.alpha*np.dot(np.dot(w,w.T),w) + self.e
        w = w * (fenzi/fenmu)

        #w = np.maximum(w,0)
        return w

    def update_h(self, m,w,h,k_s):
        fenzi = np.dot( w.T , m*self.v) + self.beta*np.dot( h , k_s )
        fenmu = np.dot(w.T,m*np.dot(w,h)+self.lammda*self.un*np.dot(w,h)) + self.beta*np.dot(np.dot(h,h.T),h) + self.e
        h = h * (fenzi/fenmu)
        
        #normalised
        #h = h/( np.sqrt( np.sum(h**2,1) ).reshape([self.k,1]) )
        #h = np.maximum(h,0)
        
        return h

    
    def cal_sparse(self,w):
        G = np.sum(w**2,1)**-0.5
        temp = np.identity(len(G))
        G = temp*G
        return G    
    
    def reconstruct_v(self,w,h):
        pre = np.dot( w , h)
        return pre
    

    
    def train(self,w,h,k_s,k_d,flag=0):
        loss_list = []
        value_list = [] 
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
        dw = np.max( np.abs(w0-w) / ( np.sqrt(np.spacing(1))+np.max(np.abs(w0)) ) )
        dh = np.max( np.abs(h0-h) / ( np.sqrt(np.spacing(1))+np.max(np.abs(h0)) ) )
        delta = np.max([dw,dh])    
        tolx = 0.001
        if delta <= tolx:
            return True
        else:
            return False
         

    

            
            
            
            
