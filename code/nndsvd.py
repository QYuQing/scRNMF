# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:16:32 2022

@author: qyq
"""
import numpy as np
def nndsvd_init_parameter(v,k,flag):
    # [1] C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head
    # start for nonnegative matrix factorization, Pattern Recognition,
    # Elsevier
    
    m = len(v)
    n = len(v[0])
    #the matrices of the factorization
    w = np.zeros([m,k])
    h = np.zeros([k,n])
    #1st SVD --> partial SVD rank-k to the input matrix A
    U,S,V = np.linalg.svd(v)
    U = U[:,:k]
    V = V[:k,:]
    #choose the first singular triplet to be nonnegative
    w[:,0] = np.sqrt(S[0])*abs(U[:,0])        
    h[0,:] = np.sqrt(S[0]) * abs(V[0,:])
        
    # 2nd SVD for the other factors (see table 1 in our paper)
    for i in range(1,k):
        uu,vv = U[:,i], V[i,:]
        uup, uun= nndsvd_posi(uu),nndsvd_nega(uu)
        vvp,vvn = nndsvd_posi(vv),nndsvd_nega(vv)
        n_uup = np.linalg.norm(uup)
        n_vvp = np.linalg.norm(vvp)
        n_uun = np.linalg.norm(uun)
        n_vvn = np.linalg.norm(vvn)
        termp = n_uup*n_vvp
        termn = n_uun*n_vvn
        if termp >= termn:
            w[:,i] = np.sqrt(S[i]*termp)*uup/n_uup
            h[i,:] = np.sqrt(S[i]*termp)*vvp.T/n_vvp
        else:
            w[:,i] = np.sqrt(S[i]*termn)*uun/n_uun
            h[i,:] = np.sqrt(S[i]*termn)*vvn.T/n_vvn
    #actually these numbers are zeros
    # w[w<0.0000000001]=0.1;
    # h[h<0.0000000001]=0.1;

    # NNDSVDa: fill in the zero elements with the average 
    if flag==1:
        ind1 = np.where(w==0)
        ind2 = np.where(h==0)
        average = np.mean(v)
        w[ind1] =  average
        h[ind2] =  average
    # NNDSVDar: fill in the zero elements with random values in the space [0:average/100]
    elif flag==2:
        ind1 = np.where(w==0)
        ind2 = np.where(h==0)
        n1 = len(ind1[0])
        n2 = len(ind2[0])
   
        average   =  np.mean(v)       ;
        w[ind1] =  (average*np.random.rand(n1)/100)
        h[ind2] =  (average*np.random.rand(n2)/100) 
    
    return w,h

def nndsvd_posi(w):
    return (w>=0)*w
def nndsvd_nega(w):
    return (w<0)*-w