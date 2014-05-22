
from glob import glob
from os.path import dirname, join
import os
from scipy.misc import imread
from masque.utils import read_landmarks
import masque


def interactive(fn):
    def wrapped(*args, **kwargs):
        if os.environ.get('INTERACTIVE'):
            return fn(*args, **kwargs)
        else:             
            print 'Ignoring interactive function: %s' % fn.__name__
    return wrapped


def data_dir_path():
    root_dir = dirname(dirname(masque.__file__))
    return join(root_dir, 'tests', 'data')


def read_images_and_landmarks(data_dir):
    image_and_landmark_paths = zip(glob(join(data_dir, 'images/*')),
                                   glob(join(data_dir, 'landmarks/*')))
    images_and_landmarks = []
    for im_path, lm_path in image_and_landmark_paths:
        im, lms = (imread(im_path), read_landmarks(lm_path))
        images_and_landmarks.append((im, lms))
    return images_and_landmarks