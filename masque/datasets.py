"""
Dataset utilities and one-time functions
"""
import os
import numpy as np
import cv2
from scipy.misc import imread


def data_dir():
    """Get default data directory path"""
    mod_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(mod_dir, '..', 'data')


def save_dataset(path, dataset):
    """Serializes and saves dataset (tuple of X and y)
    to a file"""
    assert len(dataset) == 2, 'Dataset should be tuple of (X, y)'
    np.savez(path, X=dataset[0], y=dataset[1])


def load_dataset(path):
    """Loads dataset (tuple of X and y) from pickled object"""
    npz = np.load(path)
    return npz['X'], npz['y']


def _cohn_kanade(datadir, im_shape=(256, 256), na_val=-1):
    """Creates dataset (pair of X and y) from Cohn-Kanade
    image data (CK+)"""
    images = []
    labels = []
    for name in os.listdir(os.path.join(datadir, 'faces')):
        impath = os.path.join(datadir, 'faces', name)
        labelpath = os.path.join(datadir, 'labels',
                                 name.replace('.png', '_emotion.txt'))
        # processing image
        im = imread(impath)
        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, im_shape)
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


def cohn_kanade(datadir=None, labeled_only=False, force=False):
    """
    Load Cohn-Kanade dataset. If previously loaded and force is
    set to False (default), will read data from cached file

    Params
    ------
    datadir : string, optional
        Path to CK+ data directory. This directory should already
        have 'faces' subdir.
    labeled_only : boolean, optional
        If true, only data with labels will be loaded.
        Otherwise all data will be loaded and unlabeled examples
        marked with -1 for y. Default is False.
    force : boolean, optional
        Force reloading dataset from CK+ data. Default is False.
    """
    datadir = datadir or data_dir()
    saved_dataset_file = os.path.join(datadir, 'CK.npz')
    if not force and os.path.exists(saved_dataset_file):
        X, y = load_dataset(saved_dataset_file)
    else:
        X, y = _cohn_kanade(datadir)
        save_dataset(saved_dataset_file, (X, y))
    if labeled_only:
        X = X[y != -1]
        y = y[y != -1]
    return X, y
