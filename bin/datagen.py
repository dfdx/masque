#!/usr/bin/env python

from __future__ import division
from os.path import join
import os
import argparse
import time
import glob
import numpy as np
import cv2

if '__file__' in globals():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from masque.utils import facedet
from masque.utils import read_landmarks, move_landmarks, write_landmarks

import logging
log = logging.getLogger('masque')

# def gen_face_landmarks(basedir):
#     face_files = os.listdir(os.path.join(basedir, 'faces'))
#     for face_file in face_files:
#         lm_filename = '_'.join(face_file.split('_')[:3]) + '_landmarks.txt'
#         lm_path = os.path.join(basedir, 'landmarks', lm_filename)


IMG_SUBDIR = 'images'
LM_SUBDIR = 'landmarks'
FACE_SUBDIR = 'faces'
FACE_LM_SUBDIR = 'face_landmarks'

def gen_face_data(basedir):
    """
    Generates face data: face from image and moved landmark coordinates
    """
    image_files = glob.glob(join(basedir, IMG_SUBDIR, '*'))
    face_dir = join(basedir, FACE_SUBDIR)
    face_lm_dir = join(basedir, FACE_LM_SUBDIR)
    if not os.path.exists(face_dir):
        os.mkdir(face_dir)
    if not os.path.exists(face_lm_dir):
        os.mkdir(face_lm_dir)
    for i, image_file in enumerate(image_files):
        # processing face
        im = cv2.imread(image_file)
        face_rects = facedet(im) if im != None else []
        if len(face_rects) != 1:
            # don't know exactly what to do, skipping
            continue
        i, j, h, w = face_rects[0]
        face = im[i : i+h+30, j : j+w] # extend height and crop
        face_file = join(face_dir, os.path.basename(image_file))
        cv2.imwrite(face_file, face)
        # processing landmarks
        lm_file_base = (os.path.basename(face_file).split('.')[0] + 
                        '_landmarks.txt')
        lm_file = join(basedir, LM_SUBDIR, lm_file_base)
        lms = read_landmarks(lm_file)
        face_lms = move_landmarks(lms, (i, j))
        face_lm_file = join(face_lm_dir, lm_file_base)
        write_landmarks(face_lm_file, face_lms)
        if i % 50 == 0 and i != 0:
            print('%d files processed' % i)
    print('Done.')


def main(args):
    if args.command == 'genfaces':
        if not args.data_dir:
            log.error('Please, specify base data directory with -d option')
        else:
            gen_face_data(args.data_dir)
    else:
        log.error("Don't how to perform command: %s" , args.command)


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Data Generator')
    parser.add_argument('command', help='What to do')
    parser.add_argument('-d', '--data-dir', help='Base directory for data')
    main(parser.parse_args())