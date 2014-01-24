"""
Tests for patch-based RBM
"""

from __future__ import print_function
import time
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn import datasets
import numpy as np

from masque.patch import PatchTransformer
from masque.datasets import cohn_kanade

DATA_NAME = 'ck'

############## datasets ###############

def mnist_data():
    # pylint: disable=no-member
    digits = datasets.fetch_mldata('MNIST original', data_home='~/.sk_data')
    ds_size = digits.data.shape[0]
    X = digits.data[np.random.randint(0, ds_size, 10000)].astype('float32')
    X /= 256.
    return X, None


def mnist_shapes():
    return (28, 28), (12, 12)


def ck_data():
    im_shape = ck_shapes()[0]
    XX, yy = cohn_kanade(image_shape=im_shape, basedir='data/CK')
    # X = XX[yy != -1]
    # y = yy[yy != -1]
    return XX, yy


def ck_shapes():
    return (128, 128), (12, 12)

###### common dataset wrappers #######

def get_data():
    """Get data from configured dataset"""
    return globals()[DATA_NAME + '_data']()

def get_shapes():
    """Get image shape and patch shape"""
    return globals()[DATA_NAME + '_shapes']()


########### main functions ############


def main():
    start = time.time()
    im_shape, patch_shape = get_shapes()
    n_components = patch_shape[0] * patch_shape[1] / 2
    # print(im_shape, patch_shape, n_components)
    X, _ = get_data()
    print('data shape: %s' % (X.shape,))
    model = Pipeline([
        ('patch_trans', PatchTransformer(im_shape, patch_shape, n_patches=50)),
        ('rbm', BernoulliRBM(n_components=n_components, verbose=True)),
        # ('rbm2', BernoulliRBM(n_components=n_components / 2, verbose=True)),
    ])
    model.fit(X)
    print('Time taken: %d seconds' % (time.time() - start,))
    return model
