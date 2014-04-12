"""
Playground for landmark-based experiments
"""
from __future__ import print_function
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
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


def grid_search_svc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [0.025, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['linear'],
                         'C': [0.025, 0.1, 1, 10, 100, 1000]}]
    scorings = ['accuracy']
    for scoring in scorings:
        model = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring=scoring)
        model.fit(X_test, y_test)
        print("Best parameters set found on development set:")
        print()
        print(model.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in model.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, model.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
