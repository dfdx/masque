"""
Tests for patch-based RBM
"""
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVC
from masque.playground.runners import pretrain_conv, plain_classify
from masque.rbm import GaussianBernoulliRBM
from masque.transform import PatchTransformer
from masque import datasets


_IM_SHAPE = (64, 64)

prbm_svc = {
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

_PS2 = (8, 8)
prbm_svc2 = {
    'pretrain_model' : Pipeline([
        ('patch_trans', PatchTransformer(_IM_SHAPE, _PS2, n_patches=10)),
        ('rbm', BernoulliRBM(n_components=32, verbose=True)),
    ]),
    'model' : SVC(kernel='linear', verbose=True),
    'pretrain_data' : lambda: datasets.cohn_kanade(im_shape=_IM_SHAPE),
    'data' : lambda: datasets.cohn_kanade(im_shape=_IM_SHAPE,
                                     labeled_only=True),
    'x_shape' : _IM_SHAPE,
    'filter_shape' : _PS2,
}


_PS3 = (8, 8)
gb_rbm_svc = {
    'pretrain_model' : Pipeline([
        ('patch_trans', PatchTransformer(_IM_SHAPE, _PS3, n_patches=10)),
        ('rbm', GaussianBernoulliRBM(n_components=32, verbose=True)),
    ]),
    'model' : SVC(kernel='linear', verbose=True),
    'pretrain_data' : lambda: datasets.cohn_kanade(im_shape=_IM_SHAPE),
    'data' : lambda: datasets.cohn_kanade(im_shape=_IM_SHAPE,
                                     labeled_only=True),
    'x_shape' : _IM_SHAPE,
    'filter_shape' : _PS3,
}


def sample_patches(X, n_im, n_patches, im_shape, patch_shape):
    patches = []
    idxs = np.random.randint(X.shape[0], size=n_im)
    for i in idxs:
        for j in range(n_patches):
            patches.append(get_patch(X[i].reshape(im_shape), patch_shape))
    return np.vstack(p.flatten() for p in patches)