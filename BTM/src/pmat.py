# -*- coding: utf-8 -*-
# as i have said in pvec.py.This script is used to defining some useful methods which can

import numpy as np
import math

# normalize to all entries sum to 1
def normalize(pmat=np.ones((1,1))):
    pmat = pmat.astype(float)
    eps = 1e-30
    smoother = eps * pmat.shape[0] * pmat.shape[1]
    s = pmat.sum()
    pmat = (pmat + eps)/(s + smoother)
    return pmat

# normalize elements in a row
def normr(pmat=np.ones((1,1)),c=0.0):
    pmat = pmat.astype(float)
    K = pmat.shape[0]
    for i in range(pmat.shape[0]):
        s = pmat[i].sum()
        assert(s>=0)
        pmat[i] = (pmat[i] + c)/(s + K*c)
    return pmat

# normalize elements in a col
def normc(pmat=np.ones((1,1)),c=0.0):
    pmat = pmat.astype(float)
    s = pmat.sum(axis=0)+c
    for i in range(pmat.shape[1]):
        pmat[:,i] = (pmat[:,i]+c)/s[i]
    return pmat

def add1_log(pmat=np.ones((1,1))):
    pmat = pmat.astype(float)
    pmat = np.log(pmat+1)
    return pmat

def _str(pmat=np.ones((1,1))):
    _str = ""
    for row in range(pmat.shape[0]):
        for col in range(pmat.shape[1]):
            _str += (str(pmat[row][col])+' ')
        _str += '\n'
    return _str

def write(pmat=np.ones((1,1)),pt=""):
    output = open(pt,'w')
    output.write(_str(pmat))

if __name__ == "__main__":
    pmat = np.array([[1,5,6],[2,3,5],[2,3,5]])
    pmat = add1_log(pmat)
    print(write(pmat,'../output/test'))