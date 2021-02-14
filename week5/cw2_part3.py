import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse.linalg import lsqr
from scipy.sparse import spdiags
from numpy.linalg import matrix_rank


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
D1x = spdiags(dat,diags_x,256,256).toarray()

diags_y = np.array([0])
D1y = spdiags(-mid,diags_y,256,256).toarray()

D = np.concatenate((D1x, D1y))

alpha = 0.1

y = lambda f: gaussian_filter(f,sigma)

# mult = lambda f,D: np.matmul(D,f)

# z = lambda f: gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel() + alpha*np.matmul(np.transpose(np.reshape(D,(512,256))),np.matmul(np.reshape(D,(512,256)),np.reshape(f,(256,256)))).ravel()
z = lambda f: 100*np.matmul(np.transpose(np.reshape(D,(512,256))),np.matmul(np.reshape(D,(512,256)),np.reshape(f,(256,256)))).ravel()

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

gmresOutput = gmres(A,ATg(g), x0 = f.ravel(), callback=counter)
print(counter.niter)

plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(256,256)),cmap='gray')


#d

def M_f(f):
    top = gaussian_filter(f,sigma).ravel()
    bottom = (alpha**0.5)*np.reshape(f,(256,256)).ravel()
    return np.vstack([top,bottom])

def MT_b(b):
    b1 = np.reshape(b[0:65536],(256,256))
    b2 = np.reshape(b[65536:],(256,256))
    return (gaussian_filter(b1,sigma) + np.sqrt(alpha)*b2).ravel()
    

A = LinearOperator(((256**2)*2,256**2),matvec = M_f, rmatvec = MT_b)
siz = g.size
b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1.0e-9)
print(lsqrOutput[2])

plt.figure(3)
plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

print(np.reshape(lsqrOutput[0],(256,256))-np.reshape(gmresOutput[0],(256,256)))
plt.show()

