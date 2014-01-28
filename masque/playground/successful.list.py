
# MNIST 1

def main():
    start = time.time()
    im_shape, patch_shape = get_shapes()
    n_components = patch_shape[0] * patch_shape[1] / 2
    X = get_data()
    model = Pipeline([
        ('patch_trans', PatchTransformer(im_shape, patch_shape, n_patches=10)),
        ('rbm', BernoulliRBM(n_components=n_components, verbose=True)),
    ])
    model.fit(X)
    print('Time taken: %d seconds' % (time.time() - start,))
    return model