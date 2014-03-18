"""
Playground for landmark-based experiments
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from masque import datasets
from masque.transform import PatchTransformer
from masque.rbm import GaussianBernoulliRBM
from masque.playground.runners import plain_classify, pretrain_classify


def _norm(X, y):
    """Scales data to [0..1] interval"""
    X = X.astype(np.float32) / (X.max() - X.min())
    return X, y

rbm_svc = {
    'pretrain_model' : BernoulliRBM(n_components=1024, verbose=True),
    'model' : SVC(kernel='linear', verbose=True),
    'pretrain_data' : lambda: _norm(*datasets.ck_lm_series()),
    'data' : lambda: _norm(*datasets.ck_lm_series(labeled_only=True)),
}

# TODO: sample
dbn_svc = {
    'pretrain_model' : Pipeline([        
        ('rbm0', BernoulliRBM(n_components=60, verbose=True)),
        ('rbm1', BernoulliRBM(n_components=80, verbose=True)),
        ('rbm2', BernoulliRBM(n_components=32, verbose=True)),
    ]),
    'model' : SVC(kernel='linear', verbose=True),
    'pretrain_data' : lambda: datasets.cohn_kanade_shapes(),
    'data' : lambda: datasets.cohn_kanade_shapes(labeled_only=True),
}
