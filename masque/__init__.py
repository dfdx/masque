
from __future__ import print_function
import numpy as np
import cv2
from scipy.misc import imread, imresize
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from operator import mul

from utils import facedet, smartshow
from utils import draw_points, findfiles
from pipeline import DeepPipeline
from datasets import *

IMAGE_SHAPE = (32, 32)



def makedataset(datadir, na_val=-1):    
    images = []
    labels = []
    for name in os.listdir(os.path.join(datadir, 'faces')):
        impath = os.path.join(datadir, 'faces', name)
        labelpath = os.path.join(datadir, 'labels', name.replace('_face.png',
                                                                 '_emotion.txt'))
        # processing image
        im = imread(impath)
        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)        
        im = cv2.resize(im, IMAGE_SHAPE)
        im = cv2.equalizeHist(im)
        im = im.astype(np.float32) / im.max()
        images.append(im.reshape(1, -1))
        # processing labels
        if os.path.exists(labelpath):
            with open(labelpath) as lf:
                label = int(float(lf.read().strip()))
                labels.append(label)
        else:
            labels.append(na_val)        
    return np.vstack(images), np.array(labels)

def makedbn(imshape, na_val=-1):
    N = reduce(mul, imshape)
    rbm1 = BernoulliRBM(n_components=N/2, n_iter=10,
                        learning_rate=0.05, verbose=True)
    rbm2 = BernoulliRBM(n_components=N/2, n_iter=10,
                        learning_rate=0.05, verbose=True)
    logr = LogisticRegression()
    model = DeepPipeline([('rbm1', rbm1), ('rbm2', rbm2), ('logr', logr)],
                         na_val=na_val)
    return model

    
def mklogistic(im_shape, **kwargs):
    return LogisticRegression()
    
def mkmodel(im_shape, **kwargs):
    mklogistic(im_shape, **kwargs)
    
def run():
    print('Creating dataset')
    X, y = makedataset('../data/CK')
    print('Creating model')
    model = makedbn(IMAGE_SHAPE)
    print('Fitting model')
    model.fit(X, y)
    return model

def cv_dbn():
    print('Creating dataset')
    X, y = makedataset('../data/CK', na_val=-1)
    print('Creating model')
    model = mkmodel(IMAGE_SHAPE, na_val=-1)
    print('Running Cross Validation')
    scores = cross_val_score(model, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return model, scores
    

    
def test_detector():
    im = imread('../data/manyfaces.jpg')
    face_rects = facedet(im)
    for x, y, w, h in face_rects:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 3)
    smartshow(im, False)
