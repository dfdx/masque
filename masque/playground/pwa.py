
from operator import itemgetter
import itertools as it
import random
import matplotlib.delaunay as triang
from masque.datasets import CKDataset
from masque.utils import implot, ij2xy
import pwa


def get_random_faces(datadir='data/CK', n_samples=4):
    dataset = CKDataset(datadir)
    data = [series for series in dataset.data if series.label != -1]
    random_idxs = random.sample(range(len(data)), n_samples)
    random_series = itemgetter(*random_idxs)(data)
    images = [series.images[-1] for series in random_series]
    shapes = [series.landmarks[-1] for series in random_series]
    mean_shape = data[10].landmarks[0]
    return images, shapes, mean_shape



def pwa_demo(datadir='data/CK'):
    images, shapes, mean_shape = get_random_faces(datadir)
    transformed = [pwa.warp(im, shape, mean_shape) 
                   for im, shape in zip(images, shapes)]
    # implot(zip(images, transformed)) # interlieve, not zip!
    implot(images + transformed)
    
