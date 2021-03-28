import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
import astra
from scipy.ndimage.filters import gaussian_filter

#image f
f = np.load('SLphan.npy')
# img = mpimg.imread('Cameraman256.png')
# f = np.float32(img)
forig=f
plt.figure(0)
imgplot = plt.imshow(f,cmap = 'gray')
plt.colorbar()

#image g
theta = 0.01
sigma = 2
f = gaussian_filter(f,sigma)
w,h = f.shape
noised = f 
# noised = f + theta*np.random.randn(w,h)

#Set up sinogram params
vol_geom = astra.create_vol_geom(w,h)
angles = np.linspace(0,np.pi,180,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)

#Define ID's
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(f, projector_id,  returnData=True)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('BP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = projector_id
alg_id = astra.algorithm.create(cfg)

def A(f):
    f_resh = np.reshape(f,(128,128))
    sinogram_id, sinogram = astra.create_sino(f_resh, projector_id,  returnData=True)
    A_x = sinogram
    return A_x

def AT(y):
    astra.algorithm.run(alg_id)
    f_rec = astra.data2d.get(rec_id)
    AT_y = f_rec
    return AT_y

#S function
def thresholdFunction(coeffs,tRange,tVal):
    bit = []
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

sinogram_id, sinogram = astra.create_sino(f, projector_id,  returnData=True)
g = astra.functions.add_noise_to_sino(sinogram,100)
# g = A(noised)
plt.figure(1)
imgplot = plt.imshow(g,cmap = 'gray')
plt.colorbar()

lambd = 0.0001

f = np.zeros([128,128])
def iterative(f):
    # Af_min_g = A(f - forig)
    Af_min_g = A(f) - g
    # print(np.sum(Af_min_g))
    AT_bit = AT(Af_min_g)
    f_min_bit = f - lambd*AT_bit
    # print(np.sum(f_min_bit))
    # wave = pywt.wavedec2(f_min_bit,'haar',level = 2)
    # Essed = thresholdFunction(wave,2,1)
    # inv_wave = pywt.waverec2(Essed,'haar')
    # return inv_wave
    # xmax, xmin = f_min_bit.max(), f_min_bit.min()
    # f_min_bit = (f_min_bit - xmin)/(xmax - xmin)
    # return inv_wave
    return f_min_bit

for i in range (10):
    f1 = f
    # f = iterative(f)
    
    # print(abs(np.sum(f)-np.sum(forig)))
    # if i >= 2:
    #     xmax, xmin = f.max(), f.min()
    #     f = (f - xmin)/(xmax - xmin)
    f = iterative(f)
    # print(abs(np.sum(f)-np.sum(forig)))
    print(np.sum(f1)-np.sum(f))

plt.figure(3)
plt.imshow(f,cmap = 'gray')
plt.colorbar()

plt.show()