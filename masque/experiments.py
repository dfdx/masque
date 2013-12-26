
from __future__ import print_function
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split
from skimage.filter import gabor_kernel
from operator import mul

from utils import conv2, smartshow
from datasets import cohn_kanade

IMAGE_SHAPE = (32, 32)

def init():
    global XX, X, yy, y 
    XX, yy = cohn_kanade(image_shape=IMAGE_SHAPE)
    X = X[y != -1]
    y = y[y != -1]

    
def cv_logistic():
    model = LogisticRegression()
    print('Running Cross-Validation')
    scores = cross_val_score(model, X, y, cv=10)
    print('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    return scores

def cv_dbn_l1():    
    rbm = BernoulliRBM(n_components=512, verbose=True)
    print('Fitting RBM with full dataset')
    rbm.fit(XX)
    Xt = rbm.transform(X)
    scores = []
    for k in range(10):
        print('Logistic regression, iteration #%d' % k)
        X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.1)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        scores.append(lr.score(X_test, y_test))
    scores = np.array(scores)
    print('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    return scores

def _gabor_kernels():
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels
    
def cv_dbn_l2():    
    rbm1 = BernoulliRBM(n_components=768, verbose=True)
    rbm2 = BernoulliRBM(n_components=512, verbose=True)
    print('Fitting RBM #1 with full dataset')
    rbm1.fit(XX)    
    print('Fitting RBM #2 with full dataset')
    XXt = rbm1.transform(XX)
    rbm2.fit(XXt)
    Xt = rbm1.transform(X)
    Xt = rbm2.transform(Xt)
    scores = []
    for k in range(10):
        print('Logistic regression, iteration #%d' % k)
        X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.1)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        scores.append(lr.score(X_test, y_test))
    scores = np.array(scores)
    print('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    return scores

def cv_dbn_l2_gabor():
    
    rbm1 = BernoulliRBM(n_components=512, verbose=True)
    rbm2 = BernoulliRBM(n_components=256, verbose=True)
    print('Fitting RBM #1 with full dataset')
    rbm1.fit(XX)    
    print('Fitting RBM #2 with full dataset')
    XXt = rbm1.transform(XX)
    rbm2.fit(XXt)
    Xt = rbm1.transform(X)
    Xt = rbm2.transform(Xt)
    scores = []
    for k in range(10):
        print('Logistic regression, iteration #%d' % k)
        X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.1)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        scores.append(lr.score(X_test, y_test))
    scores = np.array(scores)
    print('Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 3))
    return scores