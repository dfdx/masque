
from __future__ import print_function
import sys
import os
import glob
# import fnmatch
import heapq
import math
import time
import numpy as np
import cv2
from matplotlib import pylab as plt
from scipy.misc import imread


def conv2(im, kernel, mode='same', dst=None):
    source = im
    if mode == 'full':
        additional_rows = kernel.shape[0] - 1
        additional_cols = kernel.shape[1] - 1
        source = cv2.copyMakeBorder(im,
                           (additional_rows + 1) / 2, additional_rows / 2,
                           (additional_cols + 1) / 2, additional_cols / 2,
                           cv2.BORDER_CONSTANT, value = 0)
    anchor = (kernel.shape[1] - kernel.shape[1]/2 - 1,
              kernel.shape[0] - kernel.shape[0]/2 - 1)
    if not dst:
        dst = np.zeros(im.shape)
    fk = np.fliplr(np.flipud(kernel)).copy()
    dst = cv2.filter2D(source, -1, fk, anchor=anchor, delta=0,
                 borderType=cv2.BORDER_CONSTANT)
    if mode == 'valid':
        dst = dst[(kernel.shape[1]-1)/2 : dst.shape[1] - kernel.shape[1]/2,
                  (kernel.shape[0]-1)/2 : dst.shape[0] - kernel.shape[0]/2]
    return dst


def conv_transform(X, filters, x_shape, mode='valid'):
    """
    Transforms data matrix X by applying each filter in filters,
    flattening and stacking results

    Params
    ------
    X : array of shape (M, N)
        data matrix with M observation and N variables in each
    filters : sequence of arrays
        2D filters to apply to X
    x_shape : tuple
        shape of elements of X

    Returns
    -------
    Transformed data matrix
    """
    # filtered = (conv2(X, flt).flatten() for flt in filters)
    # TODO(a_zhabinski) optimize. One minor optimization is to
    #   preallocate Xt and copy results there directly, without storing in list
    filtered = []
    for x in X:
        x_filtered = []
        for flt in filters:
            new_x = conv2(x.reshape(x_shape), flt, mode=mode).flatten()
            x_filtered.append(new_x)
            # print(new_x)
        filtered.append(np.hstack(x_filtered))
        # print('filter applied, sleeping...')
        # time.sleep(3)
    Xt = np.vstack(filtered)
    return Xt


def implot(ims, subtitle='Images'):
    """
    Takes one image or list of images and tries to display them in a most
    convenient way.
    """
    if type(ims) == np.ndarray:
        plt.figure()
        plt.imshow(ims, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()
    else:
        ims = list(ims)
        n = len(ims)
        rows = math.floor(math.sqrt(n))
        cols = math.ceil(n / float(rows))
        plt.figure()
        for i, im in enumerate(ims):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(im, cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle(subtitle, fontsize=16)
        plt.show()

smartshow = implot

def mkfig(ims, subtitle='Images'):
    fig = plt.figure()
    if type(ims) == np.ndarray:
        fig.imshow(ims, cmap=plt.cm.gray, interpolation='nearest')
    else:
        ims = list(ims)
        n = len(ims)
        rows = math.floor(math.sqrt(n))
        cols = math.ceil(n / float(rows))
        for i, im in enumerate(ims):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(im, cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle(subtitle, fontsize=16)
    return fig


def rect_xy2ij(rect):
    return np.array([rect[1], rect[0], rect[3], rect[2]])


def facedet(im, cascade=None, cascade_xml='haarcascade_frontalface_alt2.xml'):
    """
    WARNING: returns list of rectangles as [[x, y, w, h], ...],
    i.e. xy coordinates and not ij.
    This is going to be changed in future.
    """
    if len(im.shape) != 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if not cascade:
        cascade = cv2.CascadeClassifier(cascade_xml)
    im = cv2.equalizeHist(im)
    face_rects_xy = cascade.detectMultiScale(im)
    face_rects = np.vstack([rect_xy2ij(rect_xy) for rect_xy in face_rects_xy])
    return face_rects


def rect_slice(rect):
    """
    Translates rect (as returned by facedet) to 4 points
    """
    i0, j0 = rect[:2]
    i1 = i0 + rect[2]
    j1 = j0 + rect[3]
    return [i0, j0, i1, j1]


def face_coords(face_rect):
    """
    Translates rect (as returned by facedet) to 4 points
    """
    i0, j0 = face_rect[:2]
    i1 = i0 + face_rect[2]
    j1 = j0 + face_rect[3]
    return np.array([[i0, j0], [i0, j1], [i1, j1], [i1, j0]])

def list_images(path):
    return glob.glob(path + '/*.jpg') + \
        glob.glob(path + '/*.png') + \
        glob.glob(path + '/*.gif') + \
        glob.glob(path + '/*.pgm')

def findfiles(path, regex):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, regex):
            matches.append(os.path.join(root, filename))
    return matches


def draw_points(im, points, xy=False):
    if not xy:
        points = [pt[::-1] for pt in points]   # ij to xy
    for pt in points:
        cv2.circle(im, tuple(pt), 5, (0, 255, 0), -1)
    return im

def show_points(im, points, xy=False):
    im = im.copy()
    im = draw_points(im, points, xy=False)
    smartshow(im)

def parse_coords(line):
    coords =  map(float, line.strip().split())
    coords = map(lambda x: x if x > 0 else 0, coords)  # fix negative coords
    coords = coords[::-1]   # xy to ij
    return coords

def read_landmarks(path):
    with open(path, 'r') as fin:
        lines = fin.readlines()
        points = [parse_coords(line) for line in lines]
        return np.array(points).astype(np.uint32)

def write_landmarks(path, lms):
    with open(path, 'w') as fout:
        for lm in lms:
            fout.write('\t%d\t%d\n' % (lm[0], lm[1]))


def read_label(path):
    with open(path) as fin:
        return int(float(fin.read().strip()))


def move_landmarks(landmarks, new_origin):
    """
    Moves all landmarks according to new origin.
    """
    oi, oj = new_origin
    return np.array([[lm[0] - oi, lm[1] - oj] for lm in landmarks])


def get_patch(im, shape):
    """get_patch(im, shape) -> patch

    Extracts patch of speficied shape from an image.
    """
    h, w = shape
    max_i = im.shape[0] - h
    max_j = im.shape[1] - w
    i = np.random.randint(0, max_i)
    j = np.random.randint(0, max_j)
    return im[i:i+h, j:j+w]


def normalize(X):
    """Normalize data: substract mean and divide by range"""
    X = X.astype(np.float32)
    # X = X - X.mean()
    X = X / (X.max() - X.min())
    return X

    
def most_active_points(im, flt, n=10):
    """
    Finds coordinates of n points that are actiavated the most
    by specified filter

    Params
    ------
    im : ndarray
        image to be checked
    flt : 2D-array
        filter to be applied
    n : number of most active 
    """
    # dummy implementation
    new_im = conv2(im, flt, mode='same')
    points = []
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            points.append((i, j, new_im[i, j]))
    top = heapq.nlargest(n, points, lambda t: t[2])
    return [(i, j) for i, j, val in top]
