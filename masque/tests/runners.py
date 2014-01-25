"""
Subroutines aimed to provide different learning scenarios
"""

from __future__ import print_function
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression

from masque.patch import PatchTransformer
from masque import datasets


conf_ex = {
    'pretrain_model' : Pipeline([
        ('patch_trans', PatchTransformer((128, 128), (12, 12), n_patches=20)),
        ('rbm', BernoulliRBM(n_components=72, verbose=True)),
    ]),
    'model' : LogisticRegression(),
    'pretrain_data' : datasets.cohn_kanade(),
    'data' : datasets.cohn_kanade(labeled_only=True),    
}


def pretrain_conv(conf):
    """
    Runner that:
    
    1. Fits pretrain_pipeline to pretrain_data
    2. Transforms data with pretrain_pipeline
    3. Runs cross-validation with pipeline and transformed data
    """
    
    
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