#!/usr/bin/env python

from __future__ import division
from os.path import join
import os
import argparse
import time
import glob
import numpy as np
import cv2
from scipy.misc import imread

if '__file__' in globals():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from masque.utils import facedet, standartize, standartize_landmarks
from masque.utils import read_landmarks, move_landmarks, write_landmarks
import pwa

import logging
log = logging.getLogger('masque')


IMG_SUBDIR = 'images'
LM_SUBDIR = 'landmarks'
FACE_SUBDIR = 'faces'
FACE_LM_SUBDIR = 'face_landmarks'
ALIGNED_FACE_SUBDIR = 'faces_aligned'
ALIGNED_FACE_LM_FILE = 'faces_aligned_landmarks.txt'

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
    for idx, image_file in enumerate(image_files):
        # processing face
        im = imread(image_file, False)
        face_rects = facedet(im) if im != None else []
        if len(face_rects) != 1:
            # don't know exactly what to do, skipping
            continue
        i, j, h, w = face_rects[0]
        cropped_face = im[i : i+h+30, j : j+w] # extend height and crop
        face = standartize(cropped_face)
        face_file = join(face_dir, os.path.basename(image_file))
        cv2.imwrite(face_file, face)
        # processing landmarks
        lm_file_base = (os.path.basename(face_file).split('.')[0] +
                        '_landmarks.txt')
        lm_file = join(basedir, LM_SUBDIR, lm_file_base)
        lms = read_landmarks(lm_file)
        cropped_face_lms = move_landmarks(lms, (i, j))
        face_lms = standartize_landmarks(cropped_face_lms, 
                                         cropped_face.shape[:2])
        face_lm_file = join(face_lm_dir, lm_file_base)
        write_landmarks(face_lm_file, face_lms)
        if idx % 50 == 0 and idx != 0:
            print('%d files processed' % idx)
    print('Done.')


def gen_aligned_faces(basedir):
    """
    Generates aligned face data from normal face data.
    """
    face_files = glob.glob(join(basedir, FACE_SUBDIR, '*'))
    aligned_face_dir = join(basedir, ALIGNED_FACE_SUBDIR)
    if not os.path.exists(aligned_face_dir):
        os.mkdir(aligned_face_dir)
    # get reference shape
    # ok_lm_files = glob.glob(join(basedir, FACE_LM_SUBDIR, '*01_landmarks.txt'))
    ref_face = imread(join(basedir, FACE_SUBDIR, 
                           'S011_006_00000001.png'), False)
    ref_lms = read_landmarks(join(basedir, FACE_LM_SUBDIR, 
                                  'S011_006_00000001_landmarks.txt'))
    for idx, face_file in enumerate(face_files):
        # aligning face
        face = cv2.imread(face_file, False)
        lm_file_base = (os.path.basename(face_file).split('.')[0] +
                        '_landmarks.txt')
        lms = read_landmarks(join(basedir, FACE_LM_SUBDIR, lm_file_base))
        aligned_face = pwa.warp(face, lms, ref_lms)
        aligned_face_file = join(aligned_face_dir, os.path.basename(face_file))
        cv2.imwrite(aligned_face_file, aligned_face)        
        if idx % 50 == 0 and idx != 0:
            print('%d files processed' % idx)
    print('Done.')


def main(args):
    if args.command == 'genfaces':
        if not args.data_dir:
            log.error('Please, specify base data directory with -d option')
        else:
            gen_face_data(args.data_dir)
    elif args.command == 'genaligned':
        if not args.data_dir:
            log.error('Please, specify base data directory with -d option')
        else:
            gen_aligned_faces(args.data_dir)
    else:
        log.error("Don't how to perform command: %s" , args.command)


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Data Generator')
    parser.add_argument('command', help='What to do')
    parser.add_argument('-d', '--data-dir', help='Base directory for data')
    main(parser.parse_args())