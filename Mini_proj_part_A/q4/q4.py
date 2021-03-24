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
# plt.figure(1)
# plt.imshow(arr,cmap='gray')
# plt.colorbar()
# print(np.shape(arr))

f_rec = pywt.waverec2(coeffs,'haar')
plt.figure(1)
imgplot = plt.imshow(f_rec,cmap = 'gray')
# plt.colorbar()
# print(coeffs[2].shape())

maxRange = 3
tRange = range(2)
tVal = 1

long_arr = arr.ravel()
print(np.shape(long_arr))

plt.figure(2)
plt.hist(long_arr,bins=30)

threshed = pywt.threshold(arr, tVal, 'hard')
plt.figure(3)
plt.imshow(threshed,cmap='gray')
plt.colorbar()

coeffs2 = pywt.array_to_coeffs(threshed, coeff_slices, output_format='wavedec2')
f_rec2 = pywt.waverec2(coeffs2,'haar')
plt.figure(4)
imgplot = plt.imshow(f_rec2,cmap = 'gray')


coeffsT = coeffs
def thresholdFunction(coeffs,tRange,tVal):
# implementation of the thresholding on wavelet coefficients
    for i in range(2):
        print(i)
        print(np.shape(coeffs[i][:]))

        coeffsT[i][:] == pywt.threshold(coeffs[i][:], tVal, 'hard')
        # print(coeffsT[i][:] - coeffs[i][:])


    return coeffsT

coT = thresholdFunction(coeffs,tRange,tVal)

arrT,coeff_slicesT = pywt.coeffs_to_array(coT)
plt.figure(5)
plt.imshow(arrT,cmap='gray')
plt.colorbar()


new_arr = arr
for i in range (128):
    for j in range (128):
        new_arr[i][j] = pywt.threshold(arr[i][j],tVal,'hard')
plt.imshow(new_arr,cmap='gray')

new_coeffs = pywt.array_to_coeffs(new_arr, coeff_slices, output_format='wavedec2')
f_rec2 = pywt.waverec2(new_coeffs,'haar')
plt.figure(6)
imgplot = plt.imshow(f_rec2,cmap = 'gray')

plt.show()