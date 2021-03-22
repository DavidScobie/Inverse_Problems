import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt

img = mpimg.imread('Cameraman256.png')
f = np.float32(img)
forig=f
# plt.figure(0)
# imgplot = plt.imshow(f,cmap = 'gray')
# plt.colorbar()


coeffs = pywt.wavedec2(f,'haar',level = 2)
print(len(coeffs))
print(np.shape(coeffs[0]))
print(np.shape((coeffs[1][0])))

plt.figure(0)
plt.imshow(coeffs[2][0],cmap='gray')

arr,coeff_slices = pywt.coeffs_to_array(coeffs)
plt.figure(1)
plt.imshow(arr,cmap='gray')
plt.colorbar()
print(coeff_slices)

# f_rec = pywt.waverec2(coeffs,'haar')
# plt.figure(2)
# imgplot = plt.imshow(f_rec,cmap = 'gray')
# plt.colorbar()
# print(coeffs[2].shape())

maxRange = 3
tRange = range(2)
tVal = 1

long_arr = arr.ravel()
print(np.shape(long_arr))

plt.figure(2)
plt.hist(long_arr,bins=30)

# threshed = pywt.threshold(arr, tVal, 'hard')
# plt.figure(3)
# plt.imshow(threshed,cmap='gray')
# plt.colorbar()

coeffsT = coeffs
def thresholdFunction(coeffs,tRange,tVal):
# implementation of the thresholding on wavelet coefficients
    for i in range(2):
        print('hi')
        coeffsT[i][:] == pywt.threshold(coeffs[i][:], tVal, 'hard')



    return coeffsT

coT = thresholdFunction(coeffs,tRange,tVal)

arrT,coeff_slicesT = pywt.coeffs_to_array(coT)
plt.figure(4)
plt.imshow(arrT,cmap='gray')
plt.colorbar()
plt.show()