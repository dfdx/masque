"""
Dataset utilities and one-time functions
"""
import os
import numpy as np
import cv2
from scipy.misc import imread
from sklearn import datasets as skdatasets
from masque.utils import read_landmarks, read_label
from masque.utils import normalize


def data_dir():
    """Get default data directory path"""
    mod_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(mod_dir, '..', 'data', 'CK')


def save_dataset(path, dataset):
    """Serializes and saves dataset (tuple of X and y)
    to a file"""
    # data = {'x%d' % i: datum for i, datum in enumerate(dataset)}
    np.savez(path, data=dataset)


def load_dataset(path):
    """Loads dataset (tuple of X and y) from pickled object"""
    npz = np.load(path)
    return npz['data']

def standartize(im, new_size):
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, new_size[::-1])  # NOTE: converting to XY coordinates
    im = cv2.equalizeHist(im)
    im = im.astype(np.float32) / im.max()
    return im


def _cohn_kanade(datadir, im_shape, na_val=-1):
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


def cohn_kanade(datadir=None, im_shape=(256, 256), labeled_only=False,
                force=False):
    """
    Load images and labeld from Cohn-Kanade dataset. If previously
    loaded and force is set to False (default), will read data from
    cached file

    Params
    ------
    datadir : string, optional
        Path to CK+ data directory. This directory should already
        have 'faces' subdir.
    im_shape : tuple
        Shape of images to generate or get from cache
    labeled_only : boolean, optional
        If true, only data with labels will be loaded.
        Otherwise all data will be loaded and unlabeled examples
        marked with -1 for y. Default is False.
    force : boolean, optional
        Force reloading dataset from CK+ data. Default is False.
    """
    datadir = datadir or data_dir()
    saved_dataset_file = os.path.join(datadir, 'CK_%s_%s.npz' % im_shape[:2])
    if not force and os.path.exists(saved_dataset_file):
        X, y = load_dataset(saved_dataset_file)
    else:
        X, y = _cohn_kanade(datadir, im_shape)
        save_dataset(saved_dataset_file, (X, y))
    if labeled_only:
        X = X[y != -1]
        y = y[y != -1]
    return X, y


def cohn_kanade_shapes(datadir=None, labeled_only=False, faces=True):
    """
    Loads annotations and labels from CK+ dataset

    Params
    ------
    datadir : string
        Path to CK+ data directory

    """
    datadir = datadir or data_dir()
    landmarks = []
    labels = []
    # lm_dir = 'face_landmarks' if faces else 'landmarks'
    for lm_name in os.listdir(os.path.join(datadir, 'face_landmarks')):
        label_name = lm_name.replace('.txt', '_emotion.txt')
        lm_path = os.path.join(datadir, 'face_landmarks', lm_name)
        label_path = os.path.join(datadir, 'labels', label_name)
        landmarks.append(read_landmarks(lm_path).flatten())
        if os.path.exists(label_path):
            labels.append(read_label(label_path))
        else:
            labels.append(-1)
    X, y = (np.vstack(landmarks), np.hstack(labels))
    X = normalize(X)
    if labeled_only:
        return X[y != -1], y[y != -1]
    else:
        return X, y


def _cohn_kanade_orig(datadir, im_shape, na_val=-1):
    """Creates dataset (pair of X and y) from Cohn-Kanade
    image data (CK+)"""
    images = []
    landmarks = []
    labels = []
    n = 0
    for name in os.listdir(os.path.join(datadir, 'images')):
        n += 1
        print('processed %d' % n)
        impath = os.path.join(datadir, 'images', name)
        lmname = name.replace('.png', '_landmarks.txt')
        lmpath = os.path.join(datadir, 'landmarks', lmname)
        labelname = name.replace('.png', '_emotion.txt')
        labelpath = os.path.join(datadir, 'labels', labelname)
        try:
            im = imread(impath)
        except IOError:
            continue
        im = standartize(im, im_shape)
        images.append(im.flatten())
        landmarks.append(read_landmarks(lmpath).flatten())
        # processing labels
        if os.path.exists(labelpath):
            labels.append(read_label(labelpath))
        else:
            labels.append(-1)
    return np.vstack(images), np.array(landmarks), np.array(labels)


def cohn_kanade_orig(datadir=None, im_shape=(100, 128), labeled_only=False,
                     force=False, idx=(0, 1, 2)):
    """
    Load original Cohn-Kanade dataset.

    Params
    ------
    datadir : string, optional
        Path to CK+ data directory. This directory should already
        have 'faces' subdir.
    im_shape : tuple
        Shape of images to generate or get from cache
    labeled_only : boolean, optional
        If true, only data with labels will be loaded.
        Otherwise all data will be loaded and unlabeled examples
        marked with -1 for y. Default is False.
    force : boolean, optional
        Force reloading dataset from CK+ data. Default is False.

    Returns
    -------
    images : 2D array
        Images from CK+ dataset. Each row corresponds to flattened image
    landmarks : 2D array
        Array of landmarks. Each row corresponds to flattened matrix of
        landmarks
    labels : 1D array
        Array of labels. Labels are number from 0 to 7 or -1, if there's no
        label available for corresponding image
    """
    datadir = datadir or data_dir()
    saved_dataset_file = os.path.join(datadir, 'CKorig_%s_%s.npz'
                                      % im_shape[:2])
    if not force and os.path.exists(saved_dataset_file):
        images, landmarks, labels = load_dataset(saved_dataset_file)
    else:
        images, landmarks, labels = _cohn_kanade_orig(datadir, im_shape)
        save_dataset(saved_dataset_file, (images, landmarks, labels))
    if labeled_only:
        images = images[labels != -1]
        landmarks = landmarks[labels != -1]
        labels = labels[labels != -1]
    return tuple([(images, landmarks, labels)[i] for i in idx])


def mnist():
    """MNIST dataset. Currently includes only X"""
    # TODO: add y (labels)
    # pylint: disable=no-member
    digits = skdatasets.fetch_mldata('MNIST original', data_home='~/.sk_data')
    ds_size = digits.data.shape[0]
    X = digits.data[np.random.randint(0, ds_size, 10000)].astype('float32')
    X /= 256.
    return X, None
