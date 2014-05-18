
from operator import itemgetter
import itertools as it
import random
import matplotlib.delaunay as triang
from masque.datasets import CKDataset
from masque.utils import implot
import pwa


def get_random_faces(datadir='data/CK', n_samples=8):
    dataset = CKDataset(datadir)
    data = [series for series in dataset.data if series.label != -1]
    random_idxs = random.sample(range(len(data)), n_samples)
    random_series = itemgetter(*random_idxs)(data)
    images = [series.images[-1] for series in random_series]
    shapes = [series.landmarks[-1] for series in random_series]
    mean_shape = data[10].landmarks[0]
    return images, shapes, mean_shape


def delaunay(vector):
    tri = triang.delaunay(vector[:, 0], vector[:, 1])[2]
    return tri


def transform_affine(im, shape, mean_shape):
    triangles = delaunay(shape)
    return pwa.warpTriangle(im, shape, mean_shape, triangles, im.shape)


def pwa_demo(datadir='data/CK'):
    images, shapes, mean_shape = get_random_faces(datadir)
    # transformed = [transform_affine(im, shape, mean_shape) 
    #                for im, shape in zip(images, shapes)]
    transformed = []
    for im, shape in it.izip(images, shapes):
        try:
            transformed.append(transform_affine(im, shape, mean_shape))
        except:
            pass
    # implot(zip(images, transformed)) # interlieve, not zip!
    implot(images + transformed)
    
