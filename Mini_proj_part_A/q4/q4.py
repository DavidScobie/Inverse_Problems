import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt

img = mpimg.imread('Cameraman256.png')
f = np.float32(img)
forig=f
plt.figure(0)
imgplot = plt.imshow(f,cmap = 'gray')
plt.colorbar()


coeffs = pywt.wavedec2(f,'haar',level = 7)
print(len(coeffs))
print(coeffs[0])
print(coeffs[1])

f_rec = pywt.waverec2(coeffs,'haar')
plt.figure(1)
imgplot = plt.imshow(f_rec,cmap = 'gray')
plt.colorbar()
# print(coeffs[2].shape())
plt.show()