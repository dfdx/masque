"""
Dataset utilities and one-time functions
"""
import os
import glob
import re
import itertools as it
import logging
import numpy as np
import cv2
from scipy.misc import imread
from sklearn import datasets as skdatasets
from masque.utils import read_landmarks, read_label
from masque.utils import normalize
from masque.utils import interp_list
from masque.utils import delaunay
import pwa # from pwa.so



log = logging.getLogger()


def data_dir():
    """Get default data directory path"""
    mod_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(mod_dir, '..', 'data', 'CK')


def save_dataset(path, *data):
    """Serializes and saves dataset (tuple of X and y)
    to a file"""
    np.savez(path, *data)


def load_dataset(path):
    """Loads dataset (tuple of X and y) from pickled object"""
    npz = np.load(path)
    return [npz[k] for k in sorted(npz.keys())]


def standartize(im, new_size=(128, 128)):
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, new_size[::-1])  # NOTE: converting to XY coordinates
    im = cv2.equalizeHist(im)
    im = im.astype(np.float32) / im.max()
    return im

def standartize_lms(lms, old_size, new_size=(128, 128)):
    i_scale = float(new_size[0]) / old_size[0]
    j_scale = float(new_size[1]) / old_size[1]
    lms_result = np.zeros(lms.shape)
    lms_result[:, 0] = lms[:, 0] * i_scale
    lms_result[:, 1] = lms[:, 1] * j_scale
    return lms_result.astype(np.uint32)
    


def mnist():
    """MNIST dataset. Currently includes only X"""
    # TODO: add y (labels)
    # pylint: disable=no-member
    digits = skdatasets.fetch_mldata('MNIST original', data_home='~/.sk_data')
    ds_size = digits.data.shape[0]
    X = digits.data[np.random.randint(0, ds_size, 10000)].astype('float32')
    X /= 256.
    return X, None


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
        save_dataset(saved_dataset_file, X, y)
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
    idx : sequence of ints
        Indexes of items to return: 0 stands for images, 1 - for landmarks
        and 2 - for labels
    """
    datadir = datadir or data_dir()
    saved_dataset_file = os.path.join(datadir, 'CKorig_%s_%s.npz'
                                      % im_shape[:2])
    if not force and os.path.exists(saved_dataset_file):
        images, landmarks, labels = load_dataset(saved_dataset_file)
    else:
        images, landmarks, labels = _cohn_kanade_orig(datadir, im_shape)
        save_dataset(saved_dataset_file, images, landmarks, labels)
    if labeled_only:
        images = images[labels != -1]
        landmarks = landmarks[labels != -1]
        labels = labels[labels != -1]
    return tuple([(images, landmarks, labels)[i] for i in idx])


def _parse_ck_name(name):
    subj_s, label_s, item_s = name.split('_')[:3]
    subj, label = int(subj_s[1:]), int(label_s),
    item = int(item_s.split('.')[0])
    return (subj, label, item)

def _make_ck_name(subj, label, item):
    return 'S%03d_%03d_%08d' % (subj, label, item)

def read_normalize(image_file):
    im = imread(image_file)
    return normalize(im)


def _ck_subj_ser(name):
    return _parse_ck_name(name)[:2]


class CKSeries(object):
    """
    Object that represents single images series (one expression) in
    Cohn-Kanade dataset
    """

    def __init__(self, subj, ser_id, label, landmarks, images):
        self.subj = subj
        self.ser_id = ser_id
        self.label = label
        self.landmarks = landmarks
        self.images = images

    def __repr__(self):
        return ('CKSeries(subj=%d,ser_id=%d,label=%d,n_images=%d)' %
                (self.subj, self.ser_id, self.label, len(self.images)))

    def __str__(self):
        return self.__repr__()


class CKDataset(object):

    def __init__(self, datadir=None, preprocess=standartize):
        self.data = []
        self.load(datadir, preprocess)

    def get_label(self, subj, ser_id):
        file_pattern = os.path.join(self.datadir, 'labels', 'S%03d_%03d_*.txt'
                                    % (subj, ser_id))
        files = glob.glob(file_pattern)
        if len(files) == 1:
            return read_label(files[0])
        elif len(files) == 0:
            return -1
        else:
            raise Exception("Found more than 1 label file for series: %s"
                            % files)

    def get_landmarks(self, image_paths):
        landmarks = []
        for im_path in image_paths:
            lm_path = (im_path.replace('/faces/', '/face_landmarks/')
                       .replace('.png', '.txt'))
            landmarks.append(read_landmarks(lm_path))
        return landmarks


    def load(self, datadir, preprocess):
        self.datadir = datadir or data_dir()
        image_files = sorted(os.listdir(os.path.join(self.datadir, 'faces')))
        log.info('Found %d images' % len(image_files))
        series = it.groupby(image_files, key=_ck_subj_ser)
        for k, g in series:
            subj, ser_id = k
            log.info('Processing subject %s, series %s' % (subj, ser_id))
            image_paths = [os.path.join(self.datadir, 'faces', im_file)
                           for im_file in g]
            orig_images = [imread(im_path) for im_path in image_paths]
            images = [preprocess(orig_im) for orig_im in orig_images]
            landmarks = [standartize_lms(lms, orig_images[0].shape[:2])
                         for lms in self.get_landmarks(image_paths)]
            label = self.get_label(subj, ser_id)
            self.data.append(CKSeries(subj, ser_id, label, landmarks, images))
        return series

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return 'CKDataset(n_items=%d)' % len(self.data)

    def __str__(self):
        return self.__repr__()


def ck_lm_series(datadir=None, align_to=20, labeled_only=False):
    """
    Subset of Cohn-Kanade dataset with face landmark series as X
    and labels as y
    """
    ck_dataset = CKDataset(datadir)
    X_lst = []
    y_lst = []
    for item in ck_dataset.data:
        landmarks = [lm.flatten() for lm in item.landmarks]
        landmarks = interp_list(landmarks, align_to)
        lm_image = np.vstack(landmarks)
        X_lst.append(lm_image.flatten())
        y_lst.append(item.label)
    X = np.vstack(X_lst)
    y = np.array(y_lst)
    del ck_dataset
    if labeled_only:
        return X[y != -1], y[y != -1]
    else:
        return X, y

def ck_lm_last(datadir=None, labeled_only=False):
    """
    Subset of Cohn-Kanade dataset with face landmarks expressing last
    keypoint configuration in a series as X and labels as y
    """
    ck_dataset = CKDataset(datadir)
    X_lst = []
    y_lst = []
    for item in ck_dataset.data:
        landmarks = item.landmarks[-1]
        X_lst.append(landmarks.flatten())
        y_lst.append(item.label)
    X = np.vstack(X_lst)
    y = np.array(y_lst)
    del ck_dataset
    if labeled_only:
        return X[y != -1], y[y != -1]
    else:
        return X, y


def ck_lm_pwa(datadir=None, ck_dataset=None, labeled_only=False):
    """
    Subset of Cohn-Kanade dataset with face landmarks as X1,
    faces, aligend to a mean shape, as X2, and labels as y
    """
    if not ck_dataset:
        ck_dataset = CKDataset(datadir)
    ref_lms = ck_dataset.data[10].landmarks[0]
    triangles = delaunay(ref_lms)
    X1_lst = []
    X2_lst = []
    y_lst = []
    for item in ck_dataset.data:
        landmarks = item.landmarks[-1]
        X1_lst.append(landmarks.flatten())
        orig_image = item.images[-1]
        image = pwa.warpTriangle(orig_image, item.landmarks[-1], ref_lms,
                                 triangles, orig_image.shape)
        X2_lst.append(image.flatten())
        y_lst.append(item.label)
    X1 = np.vstack(X1_lst)
    X2 = np.vstack(X2_lst)
    y = np.array(y_lst)
    del ck_dataset
    if labeled_only:
        return X1[y != -1], X2[y != -1], y[y != -1]
    else:
        return X1, X2, y


