import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

#a
img = mpimg.imread('Cameraman256.png')
f_img = np.float32(img)
plt.figure(0)
imgplot = plt.imshow(f_img,cmap = 'gray')
plt.colorbar()


#b
sigma = 2
theta = 0.01

# w,h = g.shape
# g = g + theta*np.random.randn(w,h)
# plt.figure(1)
# plt.imshow(g)
alpha = 0.1

def A(f_img):
    Af = scipy.ndimage.filters.gaussian_filter(f_img,sigma)
    return Af

#c
def ATA(f,alpha):
    y = A(f)
    z = A(y) + alpha*f
    return z


A_gmres = scipy.sparse.linalg.LinearOperator((256,256),matvec=ATA)

# gmresOutput = scipy.sparse.linalg.gmres(A_gmres, ATg)



# plt.show()
