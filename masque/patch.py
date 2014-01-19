
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from utils import get_patch

class PatchTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, im_shape, patch_size, n_patches=10):
        self.im_shape = im_shape
        if hasattr(patch_size, '__iter__'):
            self.patch_shape = patch_size
        else:
            self.patch_shape = (patch_size, patch_size)
        self.n_patches = n_patches

    def fit_transform(self, X):
        patches = []
        for x in X:
            for i in range(self.n_patches):
                patch = get_patch(x.reshape(self.im_shape), self.patch_shape)
                patches.append(patch)
        return np.vstack(patch.flatten() for patch in patches)