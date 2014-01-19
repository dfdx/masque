
from __future__ import print_function
import time
import sys
import os
import numpy as np
import cv2
from scipy.misc import imread, imresize, imsave
from utils import *

DEFAULT_IMAGE_SHAPE = (32, 32)

def makedataset(datadir, image_shape=DEFAULT_IMAGE_SHAPE, na_val=-1):    
    images = []
    labels = []
    for name in os.listdir(os.path.join(datadir, 'faces')):
        impath = os.path.join(datadir, 'faces', name)
        labelpath = os.path.join(datadir, 'labels', name.replace('_face.png',
                                                                 '_emotion.txt'))
        # processing image
        im = imread(impath)
        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)        
        im = cv2.resize(im, image_shape)
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

def cohn_kanade(image_shape):
    return makedataset('../data/CK', image_shape=image_shape)


def gen_face_landmarks(basedir):
    face_files = os.listdir(os.path.join(basedir, 'faces'))
    for face_file in face_files:
        lm_filename = '_'.join(face_file.split('_')[:3]) + '_landmarks.txt'
        lm_path = os.path.join(basedir, 'landmarks', lm_filename)

    
    

def gen_face_data(basedir):
    """
    Generates face data: face from image and moved landmark coordinates
    """
    face_file_names = os.listdir(os.path.join(basedir, 'images'))
    for i, face_file_name in enumerate(face_file_names):
        im = cv2.imread(os.path.join(basedir, 'images', face_file_name))
        face_rects = facedet(im) if im != None else []
        if len(face_rects) == 1: 
            face_rect = face_rects[0]        
            i0, j0, i1, j1 = rect_slice(face_rect)
            face = im[i0:i1, j0:j1]
            face_points = face_coords(face_rect)
            lm_file_name = face_file_name.split('.')[0] + '_landmarks.txt'
            lm_path = os.path.join(basedir, 'landmarks', lm_file_name)
            lms = read_landmarks(lm_path)
            face_lms = move_landmarks(lms, (face_rect[0], face_rect[1]))       
            face_path = os.path.join(basedir, 'faces', face_file_name)
            cv2.imwrite(face_path, face)
            face_lms_path = os.path.join(basedir, 'face_landmarks', \
                                         face_file_name.split('.')[0] + '.txt')
            write_landmarks(face_lms_path, face_lms)
        if i % 50 == 0 and i != 0:
            print('%d file processed, sleeping... ' % i, end='')
            time.sleep(20)
            print('continue ')
    print('Done.')
        
        
            
        