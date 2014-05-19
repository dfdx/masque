
from __future__ import print_function
import time
import os
import numpy as np
import cv2
from masque.utils import facedet, face_coords, rect_slice
from masque.utils import read_landmarks, move_landmarks, write_landmarks



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
    face_dir = os.path.join(basedir, 'faces')
    face_lm_dir = os.path.join(basedir, 'face_landmarks')
    if not os.path.exists(face_dir): 
        os.mkdir(face_dir)
    if not os.path.exists(face_lm_dir):
        os.mkdir(face_lm_dir)
    for i, face_file_name in enumerate(face_file_names):
        im = cv2.imread(os.path.join(basedir, 'images', face_file_name))
        face_rects = facedet(im) if im != None else []
        if len(face_rects) == 1:
            face_rect = face_rects[0]            
            face_rect = np.array([face_rect[0], face_rect[1],
                                  face_rect[2]+30, face_rect[3]])
            i0, j0, i1, j1 = rect_slice(face_rect)
            face = im[i0:i1, j0:j1]            
            lm_file_name = face_file_name.split('.')[0] + '_landmarks.txt'
            lm_path = os.path.join(basedir, 'landmarks', lm_file_name)
            lms = read_landmarks(lm_path)
            face_lms = move_landmarks(lms, (face_rect[0], face_rect[1]))
            face_path = os.path.join(face_dir, face_file_name)
            cv2.imwrite(face_path, face)
            face_lms_path = os.path.join(face_lm_dir, \
                                         face_file_name.split('.')[0] + '.txt')
            write_landmarks(face_lms_path, face_lms)
        if i % 50 == 0 and i != 0:
            print('%d files processed' % i)
    print('Done.')