
from __future__ import print_function
import numpy as np
from sklearn.pipeline import Pipeline


class DeepPipeline(Pipeline):
    """
    Special pipeline for combining labeled and unlabeled data.    
    Unlabeled data should have na.nan instead of label.
    Transformers (all estimators except for the last) are applied to all the data,
    while final estimator - to labeled data only.

    For usage example see Pipeline class
    
    Note, that this is only partial implementation. You can use safely use
    fit(), transform() and predict() methods, but some others may fail. 
    """

    def __init__(self, steps, na_val=np.nan):
        Pipeline.__init__(self, steps)
        self.na_val = na_val
    
    def fit(self, X, y=None, **fit_params):
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        self.steps[-1][-1].fit(Xt[y != self.na_val], y[y != self.na_val],
                               **fit_params)
        return self    
        