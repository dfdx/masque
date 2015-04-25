
from PIL import Image
from scipy.ndimage.interpolation import affine_transform
from numpy import *
from matplotlib import pylab as plt
from matplotlib import gridspec



# nabla_Ix = array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]])
# nabla_Iy = array([[1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 3, 3]])
# im_grad = (nabla_Ix, nabla_Iy)
# w, h = (4, 3)
N_p = 6

def imgrad(im):
    """[nabla(I_x), nabla(I_y)]"""
    if len(im.shape) != 2:
        raise Exception("Can work only with grayscale images")
    grad = [g.astype(int32) for g in gradient(im.astype(int32))]
    grad.reverse()
    return grad

    

def flatten_params(A, b):
    M = hstack([A, b.reshape((b.size, 1))])
    return M.flatten()

def structure_params(p):
    p = p.reshape(2, 3)
    return p[:, 0:2], p[:, -1]
    

def interp_im(im, y, x):
    x = asarray(x)
    y = asarray(y)
    
    x0 = floor(x).astype(int)
    x1 = x0 + 1
    y0 = floor(y).astype(int)
    y1 = y0 + 1
    
    x0 = clip(x0, 0, im.shape[1]-1);
    x1 = clip(x1, 0, im.shape[1]-1);
    y0 = clip(y0, 0, im.shape[0]-1);
    y1 = clip(y1, 0, im.shape[0]-1);
    
    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# TODO: Visualize! 

    
def quadtobox(im, dst, M):
    # Dimensions of destination image - integers, assume rectangle
    minv = amin(dst.T, axis=0)
    maxv = amax(dst.T, axis=0)

    # xg, yg = meshgrid(range(maxv[0] + 1), range(maxv[1] + 1))
    xg, yg = meshgrid(range(minv[0], maxv[0]), range(minv[1], maxv[1]))

    xy = vstack([xg.T.flatten(), yg.T.flatten()])
    xy = vstack([xy, ones((1, xy.shape[1]))])
    # Transform into source
    uv = dot(M, xy)
    
    # Remove homogeneous
    uv = uv[0:2,:].T
    
    # Sample
    xi = uv[:, 0].reshape((maxv[0] - minv[0], maxv[1] - minv[1])).T
    yi = uv[:, 1].reshape((maxv[0] - minv[0], maxv[1] - minv[1])).T
    wimg = interp_im(im, yi, xi)

    return wimg


def warp_a(im, p, dst):
    p = asarray(p).reshape(2, 3)
    M = vstack([p, [0, 0, 1]])
    M[0, 0] += 1
    M[1, 1] += 1
    wimg = quadtobox(im, dst, M)
    return wimg
    
    
    
def jacobian(nx, ny):
    jac_x = kron(array([range(0, nx)]), ones((ny, 1)))
    jac_y = kron(array([range(0, ny)]).T, ones((1, nx)))
    jac_zero = zeros((ny, nx))
    jac_one = ones((ny, nx))
    row_1 = hstack([jac_x, jac_zero, jac_y, jac_zero, jac_one, jac_zero])
    row_2 = hstack([jac_zero, jac_x, jac_zero, jac_y, jac_zero, jac_one])
    dW_dp = vstack([row_1, row_2])
    return dW_dp

    
def sd_images(dW_dp, im_grad, N_p, h, w):
    nabla_Ix, nabla_Iy = im_grad  # TODO: swap axes
    VI_dW_dp = zeros((h, w * N_p))
    for p in range(0, N_p):
        Tx = nabla_Ix * dW_dp[0:h, p * w : p * w + w]
        Ty = nabla_Iy * dW_dp[h:, p * w : p * w + w]
        VI_dW_dp[:, p * w : p * w + w] = Tx + Ty
    return VI_dW_dp

    
def sd_update(VI_dW_dp, error_im, N_p, w):
    sd_delta_p = zeros((N_p, 1))
    for p in range(N_p):
        h1 = VI_dW_dp[:, p*w : p*w + w]
        sd_delta_p[p] = sum(h1 * error_im)
    return sd_delta_p

    
def hessian(VI_dW_dp, N_p, w):
    H = zeros((N_p, N_p))
    for i in range(N_p):
        h1 = VI_dW_dp[:, i*w : i*w + w]
        for j in range(N_p):
            h2 = VI_dW_dp[:, j*w : j*w + w]
            H[i, j] = sum(h1 * h2)
    return H

def update_step(p, delta_p):
    p = p.reshape(2, 3)
    delta_p = delta_p.reshape((2, 3))
    # print '[0] p =', p
    # print '[1] delta_p = ', delta_p
    delta_M = vstack([delta_p, array([0, 0, 1])])
    delta_M[0, 0] = delta_M[0, 0] + 1
    delta_M[1, 1] = delta_M[1, 1] + 1
    # print '[2] delta_M =', delta_M
    delta_M = linalg.inv(delta_M)
    # print '[3] inv(delta_M) =', delta_M
    warp_M = vstack([p, array([0, 0, 1])])
    warp_M[0, 0] += 1
    warp_M[1, 1] += 1
    comp_M = dot(warp_M, delta_M)
    # print '[4] comp_M =', comp_M
    p = comp_M[0:2, :]
    p[0, 0] -= 1
    p[1, 1] -= 1
    return p.flatten()

        
def inv_comp(im, tmpl, n_iter=10, p_init=zeros((6,))):
    """Applies inverse compositional approach to aligning im to tmpl.
       Estimates vector of parameters p = [p_1, p_2, p_3, p_4, p_5, p_6]"""
    im = im.astype(int64)
    tmpl = tmpl.astype(int64)
    h, w = tmpl.shape
    # tmpl_pts = array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).T
    tmpl_pts = ij2xy(array([[90, 260], [90, 530], [400, 530], [400, 260]])).T
    nabla_T = imgrad(tmpl)
    dW_dp = jacobian(w, h)
    VT_dW_dp = sd_images(dW_dp, nabla_T, N_p, h, w)
    # show_sd_images(VT_dW_dp, w)
    # import time; time.sleep(5)
    H = hessian(VT_dW_dp, N_p, w)    
    H_inv = linalg.inv(H)
    warp_p = p_init.copy()
    fit_err = []
    for i in range(n_iter):
        print 'iteration %s' % i
        IWxp = warp_a(im, warp_p, tmpl_pts)
        plot_imgs([im, IWxp], ratios=[2, 1])
        plt.show()
        error_im = IWxp - tmpl
        fit_err.append(sqrt(mean(error_im * error_im)))
        print "MSE: ", fit_err[-1]
        sd_delta_p = sd_update(VT_dW_dp, error_im, N_p, w)
        delta_p = dot(H_inv, sd_delta_p)
        warp_p = update_step(warp_p, delta_p)                
    return warp_p

        
        
    
######### REPL Helpers ############

def show(im, gray=True):
    plt.figure()
    if gray:
        plt.gray() 
    plt.imshow(im)
    plt.show()


def show_pil(im, gray=None):
    Image.fromarray(uint8(im)).show()
    
    
def show_sd_images(sd_imgs, w):
    for i in xrange(6):
        show_pil(sd_imgs[:, i*w : (i + 1)*w])


def add_rect(i, j, h, w):
    plt.gca().add_patch(plt.Rectangle((j, i), w, h, fill=False))


def plot_imgs(imgs, ratios=[1, 1]):
    plt.gray()
    gs = gridspec.GridSpec(1, len(imgs), width_ratios=ratios)
    for i in range(len(imgs)):
        plt.subplot(gs[i])
        plt.imshow(imgs[i])
    return gs
    
    

######## Test Scenarios ###########

    
face_dst = array([[90, 260], [90, 530], [400, 530], [400, 260]])
    
def test_warp_a():
    im = array(Image.open('face.bmp').convert('L'))
    dst = face_dst
    p = array([0, 0, 0, 0, 0, 0])
    
        

def test_inv_comp(p_real=[0, .1, .1, 0, 0, 0], n_iter=10):
    im = asarray(Image.open('face.bmp').convert('L'))
    imh, imw = im.shape
    dst = array([[90, 260], [90, 530], [400, 530], [400, 260]])
    i0, j0 = dst.min(axis=0)
    i1, j1 = dst.max(axis=0)
    # tmpl = im[i0:i1, j0:j1]
    tmpl = warp_a(im, p_real, ij2xy(dst).T)
    return inv_comp(im, tmpl, n_iter)



def test_rect():
    im = array(Image.open('face.bmp').convert('L'))
    pts = array([[90, 260], [90, 530], [400, 530], [400, 260]])
    i0, j0 = pts.min(axis=0)
    i1, j1 = pts.max(axis=0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.gray()
    plt.imshow(im)
    add_rect(i0, j0, i1 - i0, j1 - j0)
    plt.subplot(1, 2, 2)
    box = im[i0:i1, j0:j1]
    plt.imshow(box)
    plt.show()

    
def test_rect2():
    im = array(Image.open('face.bmp').convert('L'))
    pts = array([[90, 260], [90, 530], [400, 530], [400, 260]])
    i0, j0 = pts.min(axis=0)
    i1, j1 = pts.max(axis=0)
    box = im[i0:i1, j0:j1]
    gs = plot_imgs([im, box], ratios=[2, 1])
    # plt.subplot(gs[0])
    # add_rect(i0, j0, i1 - i0, j1 - j0)
    plt.show()
    return 
    