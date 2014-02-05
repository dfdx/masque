"""
Tests for patch-based RBM
"""
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVC
from masque.playground.runners import pretrain_conv
from masque.transform import PatchTransformer


patch_rbm_svc = {
    'pretrain_model' : Pipeline([
        ('patch_trans', PatchTransformer((128, 128), (12, 12), n_patches=10)),
        ('rbm', BernoulliRBM(n_components=72, verbose=True)),
    ]),
    'model' : SVC(kernel='linear', verbose=True),
    'pretrain_data' : lambda: datasets.cohn_kanade(im_shape=(128, 128)),
    'data' : lambda: datasets.cohn_kanade(im_shape=(128, 128),
                                     labeled_only=True),
    'x_shape' : (128, 128),
    'filter_shape' : (12, 12),
}