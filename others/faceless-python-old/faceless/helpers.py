"""Helper functions used in many modules
"""
from __future__ import print_function
from matplotlib import pyplot as plt
from numpy.linalg import svd
from numpy import *

def xy2ij(shp, ni):
    out = zeros(shp.shape)
    out[:, 1] = shp[:, 0]
    out[:, 0] = ni - shp[:, 1]
    return out


def pca(data, frac=1):
    """Principal Component Analysis.
         Parameters: 
           data - data matrix with observations on rows and variables on columns
           frac - percent of variation to keep, between 0 and 1 (defaults to 1)
         Returns:
           evecs - most important principal components (eigenvectors)
           evals - corresponding eigenvalues
    """
    mn = data.mean(axis=0)        
    data = data - mn               # center data
    C = cov(data.T)                # calculate covariance matrix
    evals, evecs = linalg.eig(C)   # calculate eigenvectors and eigenvalues
    idx = argsort(evals)[::-1]     # sort both in decreasing order of evals
    evecs = evecs[:, idx]
    evals = evals[idx]
    covered = 0
    n = 0                          # number of vectors to keep
    target = sum(evals) * frac
    while covered < target:
        covered += evals[n]
        n += 1
    return evecs[:, :n], evals[:n]
    

def imshow(im, gray=True):
    plt.figure()
    if gray:
        plt.gray() 
    plt.imshow(im)
    plt.show()


    



