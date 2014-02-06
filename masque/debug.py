"""
Utilities for visual debugging of learning.
Partially based on:
  http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf
"""

import numpy as np
import matplotlib.pylab as plt
# from PIL import Image
from masque.utils import implot


def sigmoid(xx):
    return .5 * (1 + np.tanh(xx / 2.))



def plot_activ_prob(rbm, X):
    """Plots activation probability of RBM's hidden neurons for input X"""
    Xt = rbm.transform(X)
    implot(X)




# from masque import datasets
# X, y = datasets.cohn_kanade_shapes()
# from sklearn.neural_network import BernoulliRBM
