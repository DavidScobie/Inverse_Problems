import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse.linalg import lsqr


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


#c
alpha = 0.1

y = lambda f: gaussian_filter(f,sigma)

z = lambda f: gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel() + alpha*f

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

# def M_f(f):
#     top = gaussian_filter(f,sigma).ravel()
#     bottom = (alpha**0.5)*np.identity(256).ravel()
#     return np.vstack([top,bottom])

# def MT_b(b):
#     b1 = np.reshape(b[0:65536],(256,256))
#     b2 = np.reshape(b[65536:],(256,256))
#     return (gaussian_filter(b1,sigma) + np.sqrt(alpha)*b2).ravel()
#     # return gaussian_filter(np.reshape(b[0:65536],(256,256)),sigma).ravel()
    

# A = LinearOperator(((256**2)*2,256**2),matvec = M_f, rmatvec = MT_b)
# siz = g.size
# b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
# lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = (np.zeros((256,256))).ravel(),atol=1.0e-9)
# print(lsqrOutput[2])

# plt.figure(3)
# plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')
# plt.show()


# gau = lambda f: gaussian_filter(np.reshape(f,(256,256)),sigma).ravel()

# A = LinearOperator((65536,65536),matvec = gau, rmatvec = gau)

# lsqrOutput2, dummy, iter_lsqr2 = scipy.sparse.linalg.lsqr(A,g,damp=np.sqrt(alpha),atol=1.0e-9)[:3]


g = g.ravel()

gau = lambda f: gaussian_filter(np.reshape(f,(256,256)),sigma).ravel()

def M_f(f):
    # implementation of the transposed augmented matrix multiplication
    return np.append(gau(f),np.sqrt(alpha)*f)

def MT_f(f):
    # implementation of the augmented matrix multiplication
    return (gau(f[:65536]) + np.sqrt(alpha)*f[65536:])

b = np.append(g,np.zeros_like(g))

A = LinearOperator((131072,65536), matvec = M_f, rmatvec = MT_f)

lsqrOutput, dummy, iter_lsqr = scipy.sparse.linalg.lsqr(A,b,atol=1.0e-9)[:3]

print(lsqrOutput[2])

plt.figure(3)
plt.imshow(np.reshape(lsqrOutput,(256,256)),cmap='gray')
plt.show()