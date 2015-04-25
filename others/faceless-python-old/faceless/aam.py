
"""This is the main module for AAMs, containing base class `AAM`
   as long as functions to build and fit the model. Based on
   ICAAM implementation by Luca Vezzaro.
"""
from __future__ import print_function
from procrustes import procrustes
from numpy import *
from helpers import *


class AAM(object):
    def __init__(self):
        s0 = None
        shape_eiv = None
        

def build_model(shape_data, app_data, triangulation=None):
    """Builds AAM using shape and appearence data.
    """
    shape_triangles = zeros((0, 3), dtype=uint32)
    if app_data.dtype != float:
        print('Warning: appearance data not in floating point format')
        app_data = double(app_data) / 255
    if triangulation:
        shape_triangles = triangulation
    ns = shape_data.shape[0]            # numper of shapes    
    np = shape_data.shape[1]            # number of points
    nc = app_data.shape[3]              # number of colors
    # initially we use first shape instance as a mean
    mean_shape = shape_data[0, :, :]
    reference_shape = mean_shape
    aligned_data = shape_data           # matrix containing aligned shapes
    for it in range(100):
        for i in range(ns):
            d, aligned, t = procrustes(reference_shape, aligned_data[i, :, :])
            aligned_data[i, :, :] = aligned
        new_mean_shape = aligned_data.mean(axis=0)
        d, mean_shape, t = procrustes(reference_shape, new_mean_shape)
    mean_shape = aligned_data.mean(axis=0)
    # determine region of interest
    mini = mean_shape[:, 0].min()
    minj = mean_shape[:, 1].min()
    maxi = mean_shape[:, 0].max()
    maxj = mean_shape[:, 1].max()
    # place the origin in an upper left corner of bounding box
    mean_shape = mean_shape - [mini, minj] + 1
    # determine model width and height, add 1 pixel offset for gradient
    modelw = ceil(maxj - minj + 3)
    modelh = ceil(maxi - mini + 3)
    aam = AAM()
    aam.s0 = mean_shape.flatten()
    shape_matrix = aligned_data.reshape(ns, 2*np) - aam.s0
    del aligned_data
    # print(shape_matrix[0, :3])
    pc, eiv = pca(shape_matrix) 
    del shape_matrix
    aam.shape_eiv = eiv
    # Build the basis for the global shape transform, we do it here because
    # they are used to orthonormalize the shape principal vectors
    # It is done differently to the paper as we're using a different coordinate
    # frame. Here u -> i, v -> j
    s1_star = aam.s0
    
    

    

#############################################
#                SCENARIOS                  #
#############################################    

    

def run_build_model():
    import glob
    from scipy import io, misc
    DATA_DIR = '../others/icaam/examples/cootes'
    training_files = sorted(glob.glob(DATA_DIR + '/*.bmp.mat'))
    fst_im = misc.imread(training_files[0][:-4])
    ni, nj, nc = fst_im.shape
    ns = len(training_files)
    np, nd = io.loadmat(training_files[0])['annotations'].shape
    appearances = zeros((ns, ni, nj, nc))
    shapes = zeros((ns, np, nd))
    for i in range(ns):
        name = training_files[i]
        app = misc.imread(name[:-4])
        appearances[i, :, :, :] = app
        d = io.loadmat(name)
        annotations = d['annotations']
        shapes[i, :, :] = xy2ij(annotations, ni)
    AAM = build_model(shapes, appearances)
    
    
    
        
        
        
    