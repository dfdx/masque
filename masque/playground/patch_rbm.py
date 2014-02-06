"""
Tests for patch-based RBM
"""
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVC
from masque.playground.runners import pretrain_conv
from masque.transform import PatchTransformer
from masque import datasets


_IM_SHAPE = (64, 64)

patch_rbm_svc = {
    'pretrain_model' : Pipeline([
        ('patch_trans', PatchTransformer(_IM_SHAPE, (12, 12), n_patches=10)),
        ('rbm', BernoulliRBM(n_components=72, verbose=True)),
    ]),
    'model' : SVC(kernel='linear', verbose=True),
    'pretrain_data' : lambda: datasets.cohn_kanade(im_shape=_IM_SHAPE),
    'data' : lambda: datasets.cohn_kanade(im_shape=_IM_SHAPE,
                                     labeled_only=True),
    'x_shape' : _IM_SHAPE,
    'filter_shape' : (12, 12),
}