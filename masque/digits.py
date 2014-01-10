from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from utils import smartshow
import os

custom_data_home = os.getcwd() + '/sk_data'
digits = datasets.fetch_mldata('MNIST original', data_home=custom_data_home)
# ds_size = digits.data.shape[0]
# n_obs = 8192
# X = digits.data[np.random.randint(0, ds_size, n_obs)].astype('float32')

# digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
# X, Y = nudge_dataset(X, digits.target)
X, Y = X, digits.target
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

rbm = BernoulliRBM(random_state=0, verbose=True)
rbm.learning_rate = 0.06
rbm.n_iter = 20
rbm.n_components = 100
rbm.fit(X)

smartshow(c.reshape(28, 28) for c in rbm.components_[:64])
