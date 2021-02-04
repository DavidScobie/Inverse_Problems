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
theta = 0.1

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

ATg = lambda g: gaussian_filter(np.reshape(g,(256,256)),sigma).ravel()

gmresOutput = scipy.sparse.linalg.gmres(A,ATg(g), x0 = f.ravel())

plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(256,256)),cmap='gray')
# plt.show()

#d

print(np.size(A1))
iden = np.identity(256)
M_f = lambda M_f:  
MT_b = lambda MT_b: 
A1 = LinearOperator((256**2,256**2),matvec = M_f, rmatvec = MT_b)
# def M_f(A,f):
#     # implementation of the augmented matrix multiplication
#     M_f = A*f 
#     return M_f
# def MT_b(A,f):
#     # implementation of the transposed augmented matrix multiplication
#     MT_b = A.transpose * f
#     return MT_b



# siz = g.size
# b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
# lsqrOutput = scipy.sparse.linalg.lsqr(A,b)