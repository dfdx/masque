
import cv2
from numpy import *
import matplotlib.delaunay as triang
import pylab
from helpers import * 


def mean_of_columns(mat):
    """Returns 1-row matrix representing means of 
    corresponding columns 
    """
    return mat.mean(axis=0)


def center(shp):
    return mean_of_columns(shp)

    
def move_to(shp, p):
    """Moves shape so that its center is in point p
    """
    center = mean_of_columns(shp)
    # print center
    return (shp + p - center).astype(int)
        

def move_to_origin(pts):
    """Moves shape to origin and returns previous coordinates of center
    of the shape
    """
    avgs = mean_of_columns(pts)       
    for i in range(avgs.shape[0]):
        pts[:, i] = pts[:, i] - avgs[i]
    return avgs.tolist()


def dist_from_origin(pts):
    """Returns distance of every point from origin. Points should be given 
    as column matrix, where each row represents point in N-dimensional space"""
    x2 = pow(pts[:, 0], 2)
    y2 = pow(pts[:, 1], 2)
    return sqrt(x2 + y2)

    
def scale_factor(pts, pts_r):
    dist = dist_from_origin(pts)
    dist_r = dist_from_origin(pts_r)
    return mean(dist) / mean(dist_r)

    
def rotate(shp, angle): 
    rot_mat = array([[cos(angle), -sin(angle)], 
                     [sin(angle), cos(angle)]])
    shp_t = shp.transpose()
    return dot(rot_mat, shp_t).transpose()

    
def rad_to_degree(radians):
    return double(radians) * 180 / pi


def degree_to_rad(degrees):
    return double(degrees) * pi / 180


def angle_diff(a, b):
    # return arctan2(sin(a - b), cos(a - b))
    # print a, b
    d = a - b
    if d > pi:
        d -= 2 * pi
    if d < -pi:    
        d += 2 * pi
    return d

def angle_diff2(x, y):
    return [angle_diff(a, b) for a, b in zip(x.tolist(), y.tolist())]


def move_to_center(shp, size=(480, 640, 3)):
    hm = size[0] / 2
    wm = size[1] / 2
    return move_to(shp, (wm, hm))


def procrustes(shape_ref, shape):
    """ Aligns N dimentional shape represented as N-column matrix "shape" to 
    reference shape "shape_ref" of same dimentions
    
    """
    shp = copy(shape)
    shp_ref = copy(shape_ref)
    move_to_origin(shp)
    center = move_to_origin(shp_ref)
    scale = scale_factor(shp, shp_ref)
    shp *= scale    
    rot = arctan2(shp[:, 1], shp[:, 0])
    rot_r = arctan2(shp_ref[:, 1], shp_ref[:, 0])
    rot_offset = -mean(angle_diff2(rot, rot_r))
    shp = rotate(shp, rot_offset)
    shp = move_to(shp, center)
    return shp.astype(int)
    

# def procrustes(shapes):
#     shp_ref = move_to_center(shapes[0])
#     return [align(shp_ref, shp) for shp in shapes]


######### shortcuts #########
    
def show_aligned(shp_ref, shp):
    shp_aligned = align(shp_ref, shp)
    black = zeros((480, 640, 3))
    drawshape(black, shp_ref, color=(255, 0, 0), method='poly')
    drawshape(black, shp, color=(0, 255, 0), method='poly')
    drawshape(black, shp_aligned, color=(255, 255, 255), method='poly')
    show(black)


def plot_delaunay(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    cens, edg, tri, neig = triang.delaunay(x, y)
    for t in tri:
        t_i = [t[0], t[1], t[2], t[0]]
        pylab.plot(x[t_i], y[t_i])
    pylab.plot(x, y, 'o')
    pylab.show()

        
def show_pdm(shapes, size=(480, 640)):
    black = zeros(size)
    shapes = procrustes(shapes)
    for shp in shapes:
        drawshape(black, shp, pt_sz=1)
    show(black)


###########################
    
def go():
    shapes = icaam_shapes()
    show_pdm(shapes)

