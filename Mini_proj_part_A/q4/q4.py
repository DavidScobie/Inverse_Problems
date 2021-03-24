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

coeffs = pywt.wavedec2(f,'haar',level = 4)

arr,coeff_slices = pywt.coeffs_to_array(coeffs)
plt.figure(1)
plt.imshow(arr,cmap='gray')
plt.colorbar()

f_rec = pywt.waverec2(coeffs,'haar')
plt.figure(2)
imgplot = plt.imshow(f_rec,cmap = 'gray')
plt.colorbar()

bit = []
def thresholdFunction(coeffs,tRange,tVal):
    arr,coeff_slices = pywt.coeffs_to_array(coeffs)
    for i in range (tRange):
        small_bit = coeffs[i]
        bit.append(np.shape(small_bit)[1])
    new_arr = arr
    for i in range (np.sum(bit)):
        for j in range (np.sum(bit)):
            new_arr[i][j] = pywt.threshold(arr[i][j],tVal,'hard')
    coeffsT = pywt.array_to_coeffs(new_arr, coeff_slices, output_format='wavedec2')
    return coeffsT

coT = thresholdFunction(coeffs,3,1)
f_rec2 = pywt.waverec2(coT,'haar')
plt.figure(4)
plt.imshow(f_rec2,cmap = 'gray')

plt.show()