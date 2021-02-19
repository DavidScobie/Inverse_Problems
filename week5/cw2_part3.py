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
# D_2D_trans = np.transpose(D2d)
D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d

alpha = 0.016

y = lambda f: gaussian_filter(f,sigma)

# z = lambda f: gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel() + np.reshape(alpha*(DT_D@sparse.csr_matrix(np.reshape(f,(256**2,1))).toarray()),(1,256**2)).ravel()
z = lambda f: (gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel()) + (alpha*(DT_D@sparse.csr_matrix(np.reshape(f,(256**2,1))).toarray())).ravel()

A = LinearOperator((256**2,256**2),matvec = z)

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

gmresOutput = gmres(A,ATg(g), x0 = np.zeros((256,256)).ravel(), callback=counter)
# print(counter.niter)

plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(256,256)),cmap='gray')


#d

def M_f(f):
    top = gaussian_filter(f,sigma).ravel()
    middle = (alpha**0.5)*(D1x2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))).toarray().ravel()
    bottom = (alpha**0.5)*(D1y2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))).toarray().ravel()
    return np.vstack([top,middle,bottom])

def MT_b(b):
    b1 = np.reshape(b[0:65536],(256,256))
    Ag = gaussian_filter(b1,sigma)

    b2 = np.reshape(b[65536:],(2*(256**2),1))
    reg_bit = (alpha**0.5)*np.reshape((D_2D_trans@sparse.csr_matrix(b2)),(256,256)).toarray()
    return (Ag + reg_bit).ravel()
    

A = LinearOperator(((256**2)*3,256**2),matvec = M_f, rmatvec = MT_b)
siz = g.size
b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz*2,1))])
lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1.0e-3,iter_lim = 100)

print(lsqrOutput[2])
print(lsqrOutput[3])
print(lsqrOutput[8])

plt.figure(3)
plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

print(np.reshape(lsqrOutput[0],(256,256))-np.reshape(gmresOutput[0],(256,256)))
plt.show()



