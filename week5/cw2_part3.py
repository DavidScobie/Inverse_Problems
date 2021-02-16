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

mid = np.ones([1,257]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,257,256)

D1x2d = sparse.kron(scipy.sparse.identity(256),D1x)
print(D1x2d.shape)

D1y2d = sparse.kron(D1x,scipy.sparse.identity(256))
print(D1y2d.shape)

# diags_y = np.array([0])
# D1y = spdiags(-mid,diags_y,256**2,256**2)

# multip = np.matmul(np.transpose(D1x2d),D1x2d)+np.matmul(np.transpose(D1y2d),D1y2d)
lap = -((np.transpose(D1x2d)@D1x2d) + (np.transpose(D1y2d)@D1y2d))

A1 = scipy.sparse.identity(256**2)-(0.25*lap)
print(A1[500,:])
# print(A.shape)

f_sparse = sparse.csr_matrix(np.reshape(f,(256**2,1)))
print(f_sparse)

# T = np.reshape(1*(A@sparse.csr_matrix(np.reshape(f,(256**2,1))).toarray()),(1,256**2)).ravel()
# print(T)
# print(type(T))
# print(T.shape)

# y = lambda f: gaussian_filter(f,sigma)
# K = gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel()
# print(K)
# print(type(K))
# print(K.shape)

# both = T+K
# print(both)

# print(nut)

alpha = 1.5

y = lambda f: gaussian_filter(f,sigma)

z = lambda f: gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel() + np.reshape(alpha*(A1@sparse.csr_matrix(np.reshape(f,(256**2,1))).toarray()),(1,256**2)).ravel()
# z = lambda f: np.reshape(alpha*(A1@sparse.csr_matrix(np.reshape(f,(256**2,1))).toarray()),(1,256**2)).ravel()

A = LinearOperator((256**2,256**2),matvec = z)

b = np.zeros((256**2,1))

ATg = lambda g: gaussian_filter(np.reshape(g,(256,256)),sigma).ravel()

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

counter = gmres_counter()

# gmresOutput = gmres(A,ATg(g), x0 = f.ravel(), callback=counter, maxiter = 21)
gmresOutput = gmres(A,ATg(g), x0 = np.zeros((256,256)).ravel(), callback=counter, maxiter = 6)
print(counter.niter)

plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(256,256)),cmap='gray')
plt.show()


#d

# def M_f(f):
#     top = gaussian_filter(f,sigma).ravel()
#     bottom = (alpha**0.5)*np.reshape(f,(256,256)).ravel()
#     return np.vstack([top,bottom])

# def MT_b(b):
#     b1 = np.reshape(b[0:65536],(256,256))
#     b2 = np.reshape(b[65536:],(256,256))
#     return (gaussian_filter(b1,sigma) + np.sqrt(alpha)*b2).ravel()
    

# A = LinearOperator(((256**2)*2,256**2),matvec = M_f, rmatvec = MT_b)
# siz = g.size
# b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
# lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1.0e-9)
# print(lsqrOutput[2])

# plt.figure(3)
# plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

# print(np.reshape(lsqrOutput[0],(256,256))-np.reshape(gmresOutput[0],(256,256)))
# plt.show()

