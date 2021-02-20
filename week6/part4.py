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

#3

mid = np.ones([1,256]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,256,256)

D1x2d = sparse.kron(scipy.sparse.identity(256),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(256))

D2d = scipy.sparse.vstack([D1x2d,D1y2d])
D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d

T=0.2
del_X_f = D1x2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))
del_Y_f = D1y2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))

del_X_f_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_X_f),scipy.sparse.csr_matrix(del_X_f))
del_Y_f_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_Y_f),scipy.sparse.csr_matrix(del_Y_f))
sqrt_bit = scipy.sparse.csr_matrix.sqrt(del_X_f_squ + del_Y_f_squ)
exponent = -sqrt_bit/T

gamma_diag = (np.exp(exponent.todense()))

gamma_diag_array = np.ravel((gamma_diag.T).sum(axis=0))


gamma = dia_matrix((gamma_diag_array, np.array([0])), shape=(256**2, 256**2))
print(gamma)

# alpha = 0.1

# def M_f(f):
#     top = gaussian_filter(f,sigma).ravel()
#     middle = (alpha**0.5)*(D1x2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))).toarray().ravel()
#     bottom = (alpha**0.5)*(D1y2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))).toarray().ravel()
#     return np.vstack([top,middle,bottom])

# def MT_b(b):
#     b1 = np.reshape(b[0:65536],(256,256))
#     Ag = gaussian_filter(b1,sigma)

#     b2 = np.reshape(b[65536:],(2*(256**2),1))
#     reg_bit = (alpha**0.5)*np.reshape((D_2D_trans@sparse.csr_matrix(b2)),(256,256)).toarray()
#     return (Ag + reg_bit).ravel()
    

# A = LinearOperator(((256**2)*3,256**2),matvec = M_f, rmatvec = MT_b)
# siz = g.size
# b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz*2,1))])
# lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1.0e-3,iter_lim = 100)

# print(lsqrOutput[2])
# print(lsqrOutput[3])
# print(lsqrOutput[8])

# plt.figure(3)
# plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

# # print(np.reshape(lsqrOutput[0],(256,256))-np.reshape(gmresOutput[0],(256,256)))
# plt.show()