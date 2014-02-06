"""
Collection of differnet kinds Restricted Bolzmann Machines
"""

import time

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.neural_network import BernoulliRBM
from sklearn.externals.six.moves import xrange
from sklearn.utils import atleast2d_or_csr, check_arrays
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import logistic_sigmoid


class GaussianBernoulliRBM(BernoulliRBM):
    """Gaussian-Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with Gaussian visible units and
    binary hiddens.
    """
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).
        Since visible neurons are Gaussian by the nature,
        sample is equivalent to probability itself.
        Also note, that we do not use random Gaussian noise here
        as suggested by Hinton.

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = logistic_sigmoid(np.dot(h, self.components_)
                             + self.intercept_visible_)
        return p

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        return ((v - self.intercept_visible_) / 2
                - safe_sparse_dot(v, self.intercept_visible_)
                - np.log1p(np.exp(safe_sparse_dot(v, self.components_.T)
                                  + self.intercept_hidden_)).sum(axis=1))

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        v = atleast2d_or_csr(X)
        rng = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * logistic_sigmoid(fe_ - fe, log=True)
