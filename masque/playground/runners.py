"""
Subroutines aimed to provide different learning scenarios
"""
import logging
import time

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, train_test_split

from masque.transform import PatchTransformer
from masque.utils import conv_transform
from masque import datasets

log = logging.getLogger('masque')

conf_ex = {
    'pretrain_model' : Pipeline([
        ('patch_trans', PatchTransformer((128, 128), (12, 12), n_patches=10)),
        ('rbm', BernoulliRBM(n_components=72, verbose=True)),
    ]),
    'model' : SVC(kernel='linear', verbose=True),
    'pretrain_data' : lambda: datasets.cohn_kanade(im_shape=(128, 128)),
    'data' : lambda: datasets.cohn_kanade(im_shape=(128, 128),
                                     labeled_only=True),
    'x_shape' : (128, 128),
    'filter_shape' : (12, 12),
}


def pretrain_conv(conf):
    """
    Runner that:

    1. Fits pretrain_pipeline to pretrain_data
    2. Transforms data with pretrain_pipeline
    3. Runs cross-validation with pipeline and transformed data
    """
    start = time.time()
    log.info('Loading pretraining data')
    pre_X, _ = conf['pretrain_data']()
    log.info('Building and fitting pretrain model')
    pretrain_model = conf['pretrain_model']
    pretrain_model.fit(pre_X)
    del pre_X
    # assuming last step is RBM
    filters = pretrain_model.steps[-1][1].components_  
    filters = filters.reshape(len(filters), conf['filter_shape'][0],
                              conf['filter_shape'][1])
    log.info('Loading data')
    global y
    X, y = conf['data']()
    log.info('Transforming data (convolution)')
    global Xt
    Xt = conv_transform(X, filters, conf['x_shape'])
    del X
    return # TODO
    time.sleep(20)  # cool down my poor CPU
    log.info('Building and cross-validating model')
    model = conf['model']    
    scores = cross_val_score(model, Xt, y, cv=2, verbose=True)
    log.info('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    log.info('Time taken: %d seconds' % (time.time() - start,))
    return scores


def plain_pixels(conf):
    """
    Runner that:

    1. Fits pretrain_pipeline to pretrain_data
    2. Transforms data with pretrain_pipeline
    3. Runs cross-validation with pipeline and transformed data
    """
    start = time.time()
    X, y = conf['data']()    
    log.info('Building and cross-validating model')
    model = conf['model']    
    scores = cross_val_score(model, X, y, cv=10, verbose=True)
    log.info('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    log.info('Time taken: %d seconds' % (time.time() - start,))
    return scores



def test():
    pass