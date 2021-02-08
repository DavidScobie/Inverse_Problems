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
print(f.shape)
plt.figure(0)
imgplot = plt.imshow(f,cmap = 'gray')
plt.colorbar()


#b
sigma = 5
theta = 0.01

g = scipy.ndimage.filters.gaussian_filter(f,sigma)
w,h = g.shape
print(w,h)
g = g + theta*np.random.randn(w,h)
plt.figure(1)
plt.imshow(g,cmap='gray')

#c
alpha = 0.1

y = lambda f: gaussian_filter(f,sigma)

z = lambda f: gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel() + alpha*f
# z = lambda f: gaussian_filter(y(f),sigma).ravel() + alpha*f

A = LinearOperator((256**2,256**2),matvec = z)

b = np.zeros((256**2,1))

ATg = lambda g: gaussian_filter(np.reshape(g,(256,256)),sigma).ravel()

gmresOutput = gmres(A,ATg(g), x0 = f.ravel())


plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(256,256)),cmap='gray')
# plt.show()

#d

top = gaussian_filter(f,sigma).ravel()
bottom = (alpha**0.5)*np.identity(256).ravel()
print(np.vstack([top,bottom]))

def M_f(f):
    top = gaussian_filter(f,sigma).ravel()
    bottom = (alpha**0.5)*np.identity(256).ravel()
    return np.vstack([top,bottom])

# def MT_b(b):
# rmatvec = MT_b
A = LinearOperator(((256**2)*2,256**2),matvec = M_f)
print(A.shape)
siz = g.size
print(siz)
b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
# bottom = (alpha**0.5)*np.identity(256**2)
lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = f.ravel())




# print(np.size(A1))

# M_f = lambda M_f:  
# MT_b = lambda MT_b: 

# def M_f(f):
#     #implementation of the augmented matrix multiplication
#     M_f = np.array([A,(alpha**0.5)*np.identity(256)])
#     gaussian_filter(f,sigma)
#     return M_f
# def MT_b(b):
#     # implementation of the transposed augmented matrix multiplication
#     MT_b = np.array([A.transpose,(alpha**0.5)*np.identity(256)])
#     return MT_b
# A = LinearOperator((256,256),matvec = M_f, rmatvec = MT_b)


# siz = g.size
# b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
# lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = f.ravel())