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


coeffs = pywt.wavedec2(f,'haar',level = 4)
# print(len(coeffs))
# print(np.shape(coeffs[0]))
# print(np.shape((coeffs[1][0])))

# plt.figure(0)
# plt.imshow(coeffs[2][0],cmap='gray')

# arr,coeff_slices = pywt.coeffs_to_array(coeffs)
# # plt.figure(1)
# # plt.imshow(arr,cmap='gray')
# # plt.colorbar()
# # print(np.shape(arr))

# f_rec = pywt.waverec2(coeffs,'haar')
# plt.figure(1)
# imgplot = plt.imshow(f_rec,cmap = 'gray')
# # plt.colorbar()
# # print(coeffs[2].shape())

# maxRange = 3
# tRange = range(2)
# tVal = 1

# long_arr = arr.ravel()
# print(np.shape(long_arr))

# plt.figure(2)
# plt.hist(long_arr,bins=30)

# threshed = pywt.threshold(arr, tVal, 'hard')
# plt.figure(3)
# plt.imshow(threshed,cmap='gray')
# plt.colorbar()

# coeffs2 = pywt.array_to_coeffs(threshed, coeff_slices, output_format='wavedec2')
# f_rec2 = pywt.waverec2(coeffs2,'haar')
# plt.figure(4)
# imgplot = plt.imshow(f_rec2,cmap = 'gray')


# coeffsT = coeffs
# def thresholdFunction(coeffs,tRange,tVal):
# # implementation of the thresholding on wavelet coefficients
#     for i in range(2):
#         print(i)
#         print(np.shape(coeffs[i][:]))

#         coeffsT[i][:] == pywt.threshold(coeffs[i][:], tVal, 'hard')
#         # print(coeffsT[i][:] - coeffs[i][:])


#     return coeffsT

# coT = thresholdFunction(coeffs,tRange,tVal)

# arrT,coeff_slicesT = pywt.coeffs_to_array(coT)
# plt.figure(5)
# plt.imshow(arrT,cmap='gray')
# plt.colorbar()

bit = []
def thresholdFunction(coeffs,tRange,tVal):

    arr,coeff_slices = pywt.coeffs_to_array(coeffs)
    # small_arr = pywt.coeffs_to_array(coeffs[0:tRange][:])
    # small_bit = coeffs[0]

    for i in range (tRange):
        small_bit = coeffs[i]
        print(np.shape(small_bit)[1])
        bit.append(np.shape(small_bit)[1])
    print(np.sum(bit))







    # print(np.shape(small_bit)[0])
    small_arr = pywt.coeffs_to_array(coeffs[0:tRange][:])
    new_arr = arr
    # print(np.shape(small_arr))
    # print(np.shape(small_arr)[0])
    for i in range (np.sum(bit)):
        for j in range (np.sum(bit)):

    # for i in range (np.shape(small_arr)[0]):
    #     for j in range (np.shape(small_arr)[0]):
            new_arr[i][j] = pywt.threshold(arr[i][j],tVal,'hard')

    plt.figure(5)
    plt.imshow(new_arr,cmap='gray')

    coeffsT = pywt.array_to_coeffs(new_arr, coeff_slices, output_format='wavedec2')
    return coeffsT

coT = thresholdFunction(coeffs,2,1)
f_rec2 = pywt.waverec2(coT,'haar')
plt.figure(6)
plt.imshow(f_rec2,cmap = 'gray')

# f_rec2 = pywt.waverec2(coeffsT,'haar')
# plt.figure(6)
# imgplot = plt.imshow(f_rec2,cmap = 'gray')

# small_arr,rubbish = pywt.coeffs_to_array(coeffs[0:2][:])
# print(np.shape(small_arr)[0])

plt.show()