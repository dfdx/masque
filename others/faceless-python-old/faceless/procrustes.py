"""
A port of MATLAB's `procrustes` function to NumPy.
See:
    http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy/18927641

"""
from __future__ import print_function
import numpy as np

def procrustes(X, Y):
    """
    Procrustes analysis.
    -------------------

    Inputs:
    -------
    
     X : shape to be aligned to
     Y : shape to be aligned

    Outputs:
    --------

    Z : aligned shape
    d : square error between X and Z
    
    """
    wx,hx = X.shape
    wy,wy = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2).sum()
    ssY = (Y0**2).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    traceTA = s.sum()            
    b = traceTA * normX / normY
    d = 1 - traceTA**2
    Z = normX * traceTA * np.dot(Y0, T) + muX
    return Z, d
    
    
    

##################################


def run_procrustes():
    X = np.arange(10).reshape(5, 2)
    R = np.array([[1, 2], [2, 1]])
    t = np.array([3, 5])
    Y = np.dot(X, R) + t

    return procrustes(X, Y) 
    