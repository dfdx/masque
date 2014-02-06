"""
Subroutines aimed to provide different learning scenarios
"""
import logging
import time

from sklearn.cross_validation import cross_val_score, train_test_split

from masque.utils import conv_transform
from masque import datasets

log = logging.getLogger('masque')

def pretrain_conv(conf):
    """
    Pretrain/Convolve. Steps:

    1. Fit pretrain_pipeline to pretrain_data
    2. Convolve data with learned features
    3. Run cross-validation with pipeline and transformed data
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
    X, y = conf['data']()
    log.info('Transforming data (convolution)')
    Xt = conv_transform(X, filters, conf['x_shape'])
    del X
    time.sleep(20)  # cool down my poor CPU
    log.info('Building and cross-validating model')
    model = conf['model']
    scores = cross_val_score(model, Xt, y, cv=10, verbose=True)
    log.info('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    log.info('Time taken: %d seconds' % (time.time() - start,))
    return scores


def pretrain_classify(conf):
    """
    Pretrain/Classify. Steps:

    1. Fit pretrain_pipeline to pretrain_data
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
    log.info('Loading data')
    X, y = conf['data']()
    log.info('Transforming data')
    Xt = pretrain_model.transform(X)
    del X
    time.sleep(20)  # cool down my poor CPU
    log.info('Building and cross-validating model')
    model = conf['model']
    scores = cross_val_score(model, Xt, y, cv=10, verbose=True)
    log.info('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    log.info('Time taken: %d seconds' % (time.time() - start,))
    return scores

# # TODO: reserved for future tests based on patches
# def pretrain_classify(conf):
#     """
#     Pretrain/Classify. Steps:

#     1. Fit pretrain_pipeline to pretrain_data
#     2. Transforms data with pretrain_pipeline
#     3. Runs cross-validation with pipeline and transformed data
#     """
#     start = time.time()
#     log.info('Loading pretraining data')
#     pre_X, _ = conf['pretrain_data']()
#     log.info('Building and fitting pretrain model')
#     pretrain_model = conf['pretrain_model']
#     pretrain_model.fit(pre_X)
#     del pre_X
#     # assuming last step is RBM
#     log.info('Loading data')
#     X, y = conf['data']()
#     log.info('Transforming data')
#     # TODO: need to use full DBN, not just last step
#     rbm = pretrain_model.steps[-1][1]
#     Xt = rbm.transform(X)
#     del X
#     time.sleep(20)  # cool down my poor CPU
#     log.info('Building and cross-validating model')
#     model = conf['model']
#     scores = cross_val_score(model, Xt, y, cv=2, verbose=True)
#     log.info('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
#     log.info('Time taken: %d seconds' % (time.time() - start,))
#     return scores


def plain_classify(conf):
    """
    Simple classifier. Runs cross-validation with model and data from config
    """
    start = time.time()
    X, y = conf['data']()
    log.info('Building and cross-validating model')
    model = conf['model']
    scores = cross_val_score(model, X, y, cv=10, verbose=True)
    log.info('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    log.info('Time taken: %d seconds' % (time.time() - start,))
    return scores
