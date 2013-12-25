
from __future__ import print_function
import sys
import os
import numpy as np
import cv2
from scipy.misc import imread, imresize, imsave

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

# def _read_ck_image(datadir, desc):
#     subjid, setid, imno = desc
#     path = os.path.join(datadir, 'images', 'S%03d' % (subjid,), '%03d' % (setid,),
#                         'S%03d_%03d_%08d.png' % (subjid, setid, imno,))
#     return imread(path)

# def _read_ck_label(datadir, subjid, setid, imno):
#     dirpath = os.path.join(datadir, 'labels', 'S%03d' % (subjid,),
#                            '%03d' % (setid,))
#     files = os.listdir(dirpath)
#     if len(files) == 0:
#         return None
#     elif len(files) == 1:
#         path = os.path.join(dirpath, files[0])
#         with open(path, 'r') as fin:
#             text = fin.read().strip()
#             label = int(float(text))
#         return label
#     else:
#         raise RuntimeError("Don't know how to handle %d files in %s" %
#                            (len(files), dirpath))

# def _read_ck_landmarks(datadir, subjid, setid, imno):
#     path = os.path.join(datadir, 'landmarks', 'S%03d' % (subjid,),
#                         '%03d' % (setid,),
#                         'S%03d_%03d_%08d_landmarks.txt' % (subjid, setid, imno,))
#     points = []
#     with open(path, 'r') as fin:
#         for line in fin.readlines():
#             (y, x) = [int(float(n)) for n in line.strip().split()]
#             points.append((y, x))
#     return np.array(points)
    
        
    
# def read_ck_datum(datadir, subjid, setid, imno):
#     im = _read_ck_image(datadir, subjid, setid, imno)
#     label = _read_ck_label(datadir, subjid, setid, imno)                 
#     landmarks = _read_ck_landmarks(datadir, subjid, setid, imno)
#     return im, label, landmarks


# def image_files(datadir):
#     return findfiles(datadir, '*.png')

# def face_files(datadir):
#     return findfiles(os.path.join(datadir, 'faces'), '*.png')


# def is_full(datadir, im_desc):
#     '''
#     Checks if information about image is full,
#     i.e. there's corresponding landmarks and label

#     Parameters:
#     -----------
#     datadir : string
#         root of data directory
#     im_desc : tuple
#         tuple, describing image data locations
#         (subject_id, observation, image_id)
    
#     '''
#     subj_id, obs_id, image_id = im_desc
#     subj_subpath = 'S%03d' % subj_id
#     obs_subpath = '%03d' % obs_id
#     label_subpath = 'S%03d_%03d_%08d_emotion.txt' % (subj_id, obs_id, image_id)
#     label_path = os.path.join(datadir, 'labels', subj_subpath, obs_subpath,
#                               label_subpath)                    
#     return os.path.exists(label_path)

# def parse_image_filename(filename):
#     filename = os.path.basename(filename)
#     base = filename[ : filename.find('.')]
#     print(filename)
#     subj_id_str, label_str, image_id_str = base.split('_')
#     return int(subj_id_str.lstrip('S')), int(label_str), int(image_id_str)
    
    
# def list_descs(datadir, full_only=False):
#     paths = image_files(datadir)
#     descriptions = map(parse_image_filename, paths)
#     if full_only:
#         descriptions = filter(lambda desc: is_full(datadir, desc), descriptions)
#     return descriptions

# IMAGE_SHAPE = (128, 128)
    
# def get_face(im):
#     face_rects = facedet(im)
#     if len(face_rects) == 1:
#         x, y, w, h = face_rects[0]
#         face = im[y:(y+h), x:(x+w)]
#         face = imresize(face, IMAGE_SHAPE).astype(np.float32)
#         # face = face / face.max()
#         return face
#     else:
#         return None




        
# def generate_faces(datadir, skip=0):
#     import cv2
#     descs = list_descs(os.path.join(datadir, 'images'))
#     descs = descs[skip:]
#     face_descs = []
#     count = 0
#     for desc in descs:
#         count += 1
#         print('%d/%d: ' % (count, len(descs)), end='')        
#         face_file = os.path.join(datadir, 'faces', 'S%03d' % desc[0],
#                                  '%03d' % desc[1], 'S%03d_%03d_%08d_face.png' %
#                                  desc)
#         if not os.path.exists(face_file):
#             face = get_face(_read_ck_image(datadir, desc))
#             # face = _read_ck_image(datadir, desc)
#             if face is not None:
#                 face_descs.append(desc)
#                 print('Saving %s' % face_file)
#                 if not os.path.exists(os.path.dirname(face_file)):
#                     os.makedirs(os.path.dirname(face_file))
#                 # imsave(face_file, face)
#                 cv2.imwrite(face_file, face)
#             else:
#                 print("Can't detect face: %s" % face_file)
#         else:
#             print("File exists: %s" % face_file)
#     print('%d of %d faces saved' % (len(face_descs), len(descs)))
    

# def flatten_dir(dir_in, dir_out):
#     filenames = sorted(findfiles(dir_in, '*'))
#     for filename in filenames:
#         print('Copying file %s' % filename)
#         shutil.copyfile(filename, os.path.join(dir_out,
#                                                os.path.basename(filename)))
#     print('Done.')


# def flatten_rename_dir(dir_in, dir_out):
#     filenames = sorted(findfiles(dir_in, '*'))
#     for filename in filenames:
#         print('Copying file %s' % filename)
#         new_name = 'S%s_%s_%s' % tuple(filename.split('/')[-3:])
#         shutil.copyfile(filename, os.path.join(dir_out, new_name))    
#     print('Done.')


# def rename_faces(dirpath):
#     for name in os.listdir(dirpath):        
#         os.rename(os.path.join(dirpath, name), os.path.join(dirpath, name[1:]))
        