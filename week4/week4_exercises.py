import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage.filters import gaussian_filter

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

A = LinearOperator((65536,65536),matvec = z)

b = np.zeros((256**2,1))

ATg = lambda g: gaussian_filter(np.reshape(g,(256,256)),sigma).ravel()

gmresOutput = scipy.sparse.linalg.gmres(A,ATg(g), x0 = f.ravel())

plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(256,256)),cmap='gray')
plt.show()