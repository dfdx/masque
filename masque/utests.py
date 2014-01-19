
import unittest
import os
import numpy as np
from sklearn import datasets
from patch import PatchTransformer
from utils import *

class UtilTest(unittest.TestCase):

    def setUp(self):
        self.im_shape = (64, 64)
        self.im = np.random.randint(0, 256, self.im_shape).astype(np.uint8)
    
    def test_get_patch(self):
        for i in range(1000):
            shape = (np.random.randint(1, self.im_shape[0]),
                     np.random.randint(1, self.im_shape[1]))
            patch = get_patch(self.im, shape)


class PatchTest(unittest.TestCase):

    def setUp(self):        
        digits = datasets.fetch_mldata('MNIST original', data_home='~/.sk_data')
        ds_size = digits.data.shape[0]
        X = digits.data[np.random.randint(0, ds_size, 10000)].astype('float32')
        self.X = X / 256.
        self.im_shape = (28, 28)
        self.patch_size = 8

    def test_patch_transform(self):
        patch_trans = PatchTransformer(self.im_shape, self.patch_size)
        Xt = patch_trans.fit_transform(self.X)
        implot(xt.reshape(self.patch_size, self.patch_size) for xt in Xt[:64])