
import sys
import os
import glob
import heapq
import math
import time
import numpy as np
import cv2
from matplotlib import pylab as plt
from matplotlib import delaunay as triang
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


def rect_xy2ij(rect_xy):
    """
    Translates rectange from (x, y, w, h) to (i, j, h, w) by swapping 
    appropriate coordinates.
    """
    return np.array([rect_xy[1], rect_xy[0], rect_xy[3], rect_xy[2]],
                    dtype=rect_xy.dtype)
    

def facedet(im, cascade=None, cascade_xml='haarcascade_frontalface_alt2.xml'):
    """
    Detect faces on an image. 
    NOTE: one can also specify different cascade classifier or training XML
    to detect other objects.
    
    Params
    ------
    im : ndarray
        Image to detect faces on
    cascade : cv2.CascadeClassifier
        Cascade classifier to use for detection.
    cascade_xml : string
        If cascade patameter is not specified, this file is used to train one.
    
    Returns
    -------
    result : ndarray
        Rectangles, representing detected face boundaries. 
        Return data format: 
            np.array([[i0, j0, h0, w0],  # 0th face rectangle
                      [i1, j1, h1, w1],  # 1st face rectangle
                      ...])
    """
    if len(im.shape) != 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
    if not cascade:
        if os.path.exists(cascade_xml):
            cascade = cv2.CascadeClassifier(cascade_xml)
        else:
            raise IOError("Cascade training file doesn't exist: %s" 
                          % cascade_xml)
    im = cv2.equalizeHist(im)
    face_rects_xy = cascade.detectMultiScale(im)    
    face_rects = np.vstack([rect_xy2ij(rect_xy) for rect_xy in face_rects_xy])
    return face_rects


def draw_landmarks(im, points, orientation='ij'):
    if orientation == 'ij':
        points = [pt[::-1] for pt in points]   # ij to xy
    elif orientation == 'xy':
        pass
    else: 
        raise RuntimeError("Unkown landmark orientation type: %s" % orientation)
    for pt in points:
        cv2.circle(im, tuple(pt), 2, (0, 255, 0), -1)
    return im


def plot_landmarks(im, points, orientation='ij'):
    im = im.copy()
    im = draw_landmarks(im, points, orientation)
    implot(im)


def delaunay(landmarks):
    tri = triang.delaunay(landmarks[:, 0], landmarks[:, 1])[2]
    return tri


def draw_tri(im, lms, tri, copy=True, color=(0, 255, 0)):
    """
    Plot triangualtion

    Params: 
    im : ndarray
        Image to draw on
    lms : list of tuples or ndarray
        Original landmarks
    tri : ndarray
        Trianguation, ndarray of shape Nx3, where each row shows indicies 
        or triangle corners
    """
    if copy: 
        im = im.copy()
    for tr in tri: 
        i0, j0 = lms[tr[0]]
        i1, j1 = lms[tr[1]]
        i2, j2 = lms[tr[2]]
        cv2.line(im, (j0, i0), (j1, i1), color)
        cv2.line(im, (j1, i1), (j2, i2), color)
        cv2.line(im, (j2, i2), (j0, i0), color)
    return im
    
    

def _parse_coords(line):
    coords =  map(float, line.strip().split())
    coords = map(lambda x: x if x > 0 else 0, coords)  # fix negative coords    
    return coords


def read_landmarks(path, orientation='xy'):
    """
    Read landmarks form path. 
    Standard orientation is ij, if coords in file are stored as xy, 
    orientation should be set to 'xy' (this is default behavior).
    """
    with open(path, 'r') as fin:
        lines = fin.readlines()
        lms = [_parse_coords(line) for line in lines]
        lms = np.array(lms).astype(np.uint32)
        if orientation == 'xy':
            lms[:, [0, 1]] = lms[:, [1, 0]]
        return lms


def write_landmarks(path, lms, orientation='xy'):
    """
    Write landmarks to path. 
    Standard orientation is ij, to store coordinates as xy, 
    one should be set orientation to 'xy' (this is default behavior).
    """
    if orientation == 'xy':
        lms = lms.copy()
        lms[:, [0, 1]] = lms[:, [1, 0]]
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


# def normalize(X):
#     """Normalize data: substract mean and divide by range"""
#     X = X.astype(np.float32)
#     # X = X - X.mean()
#     X = X / (X.max() - X.min())
#     return X


# def most_active_points(im, flt, n=10):
#     """
#     Finds coordinates of n points that are actiavated the most
#     by specified filter

#     Params
#     ------
#     im : ndarray
#         image to be checked
#     flt : 2D-array
#         filter to be applied
#     n : number of most active
#     """
#     # dummy implementation
#     new_im = conv2(im, flt, mode='same')
#     points = []
#     for i in range(new_im.shape[0]):
#         for j in range(new_im.shape[1]):
#             points.append((i, j, new_im[i, j]))
#     top = heapq.nlargest(n, points, lambda t: t[2])
#     return [(i, j) for i, j, val in top]


# def interp_list(lst, new_length):
#     """
#     'Shrinks' or 'stratches' lst by removing or replicating elements
#     """
#     k = float(len(lst)) / new_length
#     new_lst = [None] * new_length
#     for i in range(new_length):
#         j = int(i * k)
#         new_lst[i] = lst[j]
#     return new_lst
