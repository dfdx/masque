
from __future__ import print_function
import sys
import os
import glob
import fnmatch
import math
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
        dst = dst[(kernel.shape[1]-1)/2 : dst.shape[1] - kernel.shape[1]/2, \
                  (kernel.shape[0]-1)/2 : dst.shape[0] - kernel.shape[0]/2]
    return dst


def smartshow(ims, subtitle='Images'):
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


def facedet(im, cascade=None, cascade_xml='haarcascade_frontalface_alt2.xml'):
    if len(im.shape) != 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if not cascade:
        cascade = cv2.CascadeClassifier(cascade_xml)
    im = cv2.equalizeHist(im)
    face_rects = cascade.detectMultiScale(im)
    return face_rects
    
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


def draw_points(im, points):
    for pt in points:
        cv2.circle(im, tuple(pt), 5, (0, 255, 0), -1)
    return im

def show_points(im, points):
    im = im.copy()
    im = draw_points(im, points)
    smartshow(im)

    