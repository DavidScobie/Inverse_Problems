import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse.linalg import lsqr
from scipy.sparse import spdiags
from scipy import sparse
from scipy.sparse import vstack
from scipy.sparse.linalg import expm
from scipy.sparse import dia_matrix
from scipy import stats 

#a
img = mpimg.imread('Cameraman256.png')
f = np.float32(img)
plt.figure(0)
imgplot = plt.imshow(f,cmap = 'gray')
plt.colorbar()

#b
sigma = 5
theta = 0.01

g = scipy.ndimage.filters.gaussian_filter(f,sigma)
w,h = g.shape
g = g + theta*np.random.randn(w,h)
plt.figure(1)
plt.imshow(g,cmap='gray')

#Constructing D

mid = np.ones([1,256]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,256,256)

D1x2d = sparse.kron(scipy.sparse.identity(256),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(256))

D2d = scipy.sparse.vstack([D1x2d,D1y2d])
D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d

#Finding gamma fuuction
T=0.27
# del_X_g = D1x2d@sparse.csr_matrix(np.reshape(g,(256**2,1)))
# del_Y_g = D1y2d@sparse.csr_matrix(np.reshape(g,(256**2,1)))

# del_X_g_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_X_g),scipy.sparse.csr_matrix(del_X_g))
# del_Y_g_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_Y_g),scipy.sparse.csr_matrix(del_Y_g))
# sqrt_bit = scipy.sparse.csr_matrix.sqrt(del_X_g_squ + del_Y_g_squ)
# exponent = -sqrt_bit/T
# gamma_diag = (np.exp(exponent.todense()))
# gamma_diag_array = np.ravel((gamma_diag.T).sum(axis=0))
# gamma = dia_matrix((gamma_diag_array, np.array([0])), shape=(256**2, 256**2))

# #plotting gamma values
# plt.figure(2)
# plt.imshow(np.reshape(gamma_diag,(256,256)),cmap='gray')
# plt.colorbar()

# sqrt_gam = scipy.sparse.csr_matrix.sqrt(gamma)
# sqrt_gam_dx = sqrt_gam@D1x2d
# sqrt_gam_dy = sqrt_gam@D1y2d
# sqrt_gam_D = scipy.sparse.vstack([sqrt_gam_dx,sqrt_gam_dy])
# sqrt_gam_D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(sqrt_gam_D))

alpha = 0.016

def M_f(fi):
    del_X_fi = D1x2d@sparse.csr_matrix(np.reshape(fi,(256**2,1)))
    del_Y_fi = D1y2d@sparse.csr_matrix(np.reshape(fi,(256**2,1)))

    del_X_fi_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_X_fi),scipy.sparse.csr_matrix(del_X_fi))
    del_Y_fi_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_Y_fi),scipy.sparse.csr_matrix(del_Y_fi))
    sqrt_bit = scipy.sparse.csr_matrix.sqrt(del_X_fi_squ + del_Y_fi_squ)
    exponent = -sqrt_bit/T
    gamma_diag = (np.exp(exponent.todense()))
    gamma_diag_array = np.ravel((gamma_diag.T).sum(axis=0))
    gamma = dia_matrix((gamma_diag_array, np.array([0])), shape=(256**2, 256**2))
    sqrt_gam = scipy.sparse.csr_matrix.sqrt(gamma)
    sqrt_gam_dx = sqrt_gam@D1x2d
    sqrt_gam_dy = sqrt_gam@D1y2d
    
    top = gaussian_filter(g,sigma).ravel()
    middle = (alpha**0.5)*(sqrt_gam_dx@sparse.csr_matrix(np.reshape(g,(256**2,1)))).toarray().ravel()
    bottom = (alpha**0.5)*(sqrt_gam_dy@sparse.csr_matrix(np.reshape(g,(256**2,1)))).toarray().ravel()
    return np.vstack([top,middle,bottom])

def MT_b(bi):    
    b1 = np.reshape(bi[0:65536],(256,256))
    Ag = gaussian_filter(b1,sigma)
    
    b2 = np.reshape(bi[65536:],(2*(256**2),1))

    del_X_bi = D1x2d@sparse.csr_matrix(np.reshape(b2,(256**2,2)))
    del_Y_bi = D1y2d@sparse.csr_matrix(np.reshape(b2,(256**2,2)))

    del_X_bi_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_X_bi),scipy.sparse.csr_matrix(del_X_bi))
    del_Y_bi_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_Y_bi),scipy.sparse.csr_matrix(del_Y_bi))   

    sum_cols_X = scipy.sparse.csr_matrix.sum(del_X_bi_squ,axis=1)
    sum_cols_Y = scipy.sparse.csr_matrix.sum(del_Y_bi_squ,axis=1)
    add = sparse.csr_matrix(sum_cols_X + sum_cols_Y)
    sqrt_bit = scipy.sparse.csr_matrix.sqrt(add)
    exponent = sparse.csr_matrix(-sqrt_bit/T)

    gamma_diag = (np.exp(exponent.todense()))
    gamma_diag_array = np.ravel((gamma_diag.T).sum(axis=0))
    gamma = dia_matrix((gamma_diag_array, np.array([0])), shape=(256**2, 256**2))
    sqrt_gam = scipy.sparse.csr_matrix.sqrt(gamma)
    sqrt_gam_dx = sqrt_gam@D1x2d
    sqrt_gam_dy = sqrt_gam@D1y2d    
    sqrt_gam_D = scipy.sparse.vstack([sqrt_gam_dx,sqrt_gam_dy])
    sqrt_gam_D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(sqrt_gam_D))
    
    reg_bit = (alpha**0.5)*np.reshape((sqrt_gam_D_trans@sparse.csr_matrix(b2)),(256,256)).toarray()
    return (Ag + reg_bit).ravel()
    

A = LinearOperator(((256**2)*3,256**2),matvec = M_f, rmatvec = MT_b)
siz = g.size
b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz*2,1))])
# lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1.0e-3,iter_lim = 5)
lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = g.ravel(), iter_lim = 20)


# print(lsqrOutput[2])
# print(lsqrOutput[3])
# print(lsqrOutput[8])

plt.figure(3)
plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

plt.show()