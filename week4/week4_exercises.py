import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage

#a
img = mpimg.imread('Cameraman256.png')
f_img = np.float32(img)
plt.figure(0)
imgplot = plt.imshow(f_img,cmap = 'gray')
plt.colorbar()


#b
sigma = 2
theta = 0.01
g = scipy.ndimage.filters.gaussian_filter(f_img,sigma)
w,h = g.shape
g = g + theta*np.random.randn(w,h)
plt.figure(1)
plt.imshow(g)

plt.show()
