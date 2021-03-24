import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
import astra
from scipy.ndimage.filters import gaussian_filter

#image f
img = mpimg.imread('Cameraman256.png')
f = np.float32(img)
forig=f
plt.figure(0)
imgplot = plt.imshow(f,cmap = 'gray')
plt.colorbar()

#image g
theta = 0.01
sigma = 2
f = gaussian_filter(f,sigma)
w,h = f.shape
g = f + theta*np.random.randn(w,h)
plt.figure(1)
imgplot = plt.imshow(g,cmap = 'gray')
plt.colorbar()

#Set up sinogram params
vol_geom = astra.create_vol_geom(w,h)
angles = np.linspace(0,np.pi,180,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)

#Define ID's
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(f, projector_id,  returnData=True)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = projector_id
alg_id = astra.algorithm.create(cfg)

def A(f):
    f_resh = np.reshape(f,(256,256))
    sinogram_id, sinogram = astra.create_sino(f_resh, projector_id,  returnData=True)
    A_x = sinogram
    return A_x

def AT(y):
    astra.algorithm.run(alg_id)
    f_rec = astra.data2d.get(rec_id)
    AT_y = f_rec
    return AT_y.ravel()

#S function
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

Af_min_g = A(f - g)
plt.figure(2)
plt.imshow(f - 0.001*np.reshape(AT(Af_min_g),(256,256)),cmap = 'gray')

plt.show()