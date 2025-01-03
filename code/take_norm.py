# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:09:16 2022

@author: qyq
"""
import numpy as np
def take_norm(data, cellwise_norm=True, log1p=True):
    data_norm = data.copy()
    data_norm = data_norm.astype('float')
    if cellwise_norm:
        libs = data.sum(axis=1)
        norm_factor = np.diag(np.median(libs) / libs)
        data_norm = np.dot(norm_factor, data_norm)
    data_norm = data_norm.astype('float')
    if log1p:
        data_norm = np.log(data_norm + 1.)
    return data_norm,norm_factor

def norm_return(data_norm,norm_factor):
    '''

    Parameters
    ----------
    data_norm : np
        data->lognormlized->data_norm
    norm_factor : np
        normlized
    Returns
    -------
    data

    '''
    data_norm = np.exp(data_norm)-1
    data = np.dot(np.linalg.inv(norm_factor),data_norm)
    return data
    