
from __future__ import division
import numpy as np
import glob
import os
from scipy.misc import imread
import cv2
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter 
import logging

log = logging.getLogger()

def read_label(label_file):
    with open(label_file) as fl:
        return int(float(fl.read().strip()))


def load(start, stop, datadir='data/CK'):
    im_list = glob.glob(os.path.join(datadir, 'faces_aligned/*.png'))[start:]
    if not im_list:
        msg = ('No image files found in: %s' 
               % os.path.realpath(os.path.join(datadir, 'faces_aligned')))
        log.error(msg)
        raise RuntimeException(msg)
    X = []
    y = []
    more_to_read = stop - start
    for im_file in im_list:
        if more_to_read <= 0:
            break
        label_base_pat = os.path.basename(im_file)[:9] + '*_emotion.txt'
        maybe_label_file = glob.glob(os.path.join(datadir, 'labels',
                                                  label_base_pat))
        if maybe_label_file:            
            y.append(read_label(maybe_label_file[0]))
            imdata = imread(im_file, False)            
            imdata = cv2.resize(imdata, (32, 32))
            imdata = imdata.flatten().astype(np.float32) / 255
            X.append(imdata)
            more_to_read -= 1
    return DenseDesignMatrix(X=np.asarray(X), y=np.asarray(y).reshape(-1, 1),
                             view_converter=DefaultViewConverter((32, 32, 1),
                                                             axes=('b', 0, 1, 'c')))
