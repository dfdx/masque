
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.misc import imresize
from utils import get_patch

class PatchTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms each observation (image) into a set of randomly selected patches
    """

    def __init__(self, im_shape, patch_size, n_patches=10):
        self.im_shape = im_shape
        if hasattr(patch_size, '__iter__'):
            self.patch_shape = patch_size
        else:
            self.patch_shape = (patch_size, patch_size)
        self.n_patches = n_patches

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        patches = []
        for x in X:
            for i in range(self.n_patches):
                patch = get_patch(x.reshape(self.im_shape), self.patch_shape)
                patches.append(patch)
        return np.vstack(patch.flatten() for patch in patches)

    def fit_transform(self, X, y=None):
        return self.transform(X, y)


class ResizeTransformer(BaseEstimator, TransformerMixin):
    """
    Resizes each element of datasets from size_in to size_out
    """

    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out

    def _flat_resize(self, x):
        """Image resize that works on flattened images. Grayscale images only"""
        x_im = x.reshape(self.size_in)
        new_x_im = imresize(x_im, self.size_out)
        return new_x_im.flatten()

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        return np.vstack(self._flat_resize(x) for x in X)
