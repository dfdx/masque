
from __future__ import division
import random
import unittest
import pwa
from masque.utils import implot, delaunay, draw_tri
from tests.common import read_images_and_landmarks
from tests.common import interactive, data_dir_path

class PWATest(unittest.TestCase):

    def setUp(self):
        self.data_dir = data_dir_path()

    def tearDown(self):
        pass

    def test_basic_pwa(self):
        data_tuples = read_images_and_landmarks(self.data_dir)
        im0, lms0 = data_tuples[0]
        tri0 = delaunay(lms0)
        im0 = draw_tri(im0, lms0, tri0)        
        im1, lms1 = data_tuples[1]
        tri1 = delaunay(lms1)
        im1 = draw_tri(im1, lms1, tri1)

        im1to0 = pwa.warpTriangle(im1, lms1, lms0,
                                 tri1, im0.shape)
        implot([im0, im1, im1to0])