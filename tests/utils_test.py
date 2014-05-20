
from __future__ import division
from glob import glob
from os.path import dirname, join
import os
import random
import unittest
from scipy.misc import imread
import cv2
from masque.utils import implot, delaunay, draw_tri, draw_landmarks
from masque.utils import read_landmarks, write_landmarks
from masque.utils import facedet
import masque
from tests.common import interactive


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


class LandmarksTestCase(unittest.TestCase):

    def setUp(self):
        self.data_dir = data_dir_path()

    def tearDown(self):
        pass

    @interactive
    def test_draw_landmarks(self):
        images_and_landmarks = read_images_and_landmarks(self.data_dir)[:4]
        marked_images = [draw_landmarks(im, lm)
                         for im, lm in images_and_landmarks]
        implot(marked_images)

    @interactive
    def test_delaunay(self):
        im, lms = read_images_and_landmarks(self.data_dir)[0]
        tri = delaunay(lms)
        implot(draw_tri(im, lms, tri))

    def test_lm_read_write(self):
        lm_file_path = glob(join(self.data_dir, 'landmarks/*'))[0]
        lms = read_landmarks(lm_file_path)
        lm_tmp_file_path = '/tmp/test_landmarks.txt'
        write_landmarks(lm_tmp_file_path, lms)
        try:
            lms_tmp = read_landmarks(lm_tmp_file_path)
            self.assertTrue((lms == lms_tmp).all())
        finally:
            os.remove(lm_tmp_file_path)

    @interactive
    def test_facedet(self):
        im_file_path = glob(join(self.data_dir, 'images/*'))[0]
        im = imread(im_file_path)
        rects = facedet(im)
        for i, j, h, w in rects:
            cv2.rectangle(im, (j, i), (j+w, i+h), 
                          color=(0, 255, 0), thickness=3)
        implot(im)
