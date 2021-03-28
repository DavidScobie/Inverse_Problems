import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
import astra
from scipy.ndimage.filters import gaussian_filter

#load phantom in
f = np.load('SLphan.npy')
forig = f

#Set up sinogram params
w,h = f.shape
vol_geom = astra.create_vol_geom(w,h)
angles = np.linspace(0,np.pi,180,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)

projector_id = astra.create_projector('strip', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(f, projector_id,  returnData=True)

#g is a noisy sinogram of f
g = astra.functions.add_noise_to_sino(sinogram,100)

# plt.figure(0)
# plt.imshow(g)

# plt.figure(1)
# plt.imshow(sinogram)

#Af_min_g is just noise
Af_min_g = sinogram - g
# plt.figure(2)
# plt.imshow(Af_min_g)

#Set up params for AT of Af_min_g
Af_min_g_id = astra.data2d.create('-sino',proj_geom,Af_min_g)
cfg = astra.astra_dict('BP')
rec_id = astra.data2d.create('-vol', vol_geom)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = Af_min_g_id
cfg['ProjectorId'] = projector_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
AT_Af_min_g = astra.data2d.get(rec_id)

#AT(Af_min_g) is also just noise
# plt.figure(3)
# plt.imshow(AT_Af_min_g)

#define lambda
lambd = 0.00005
mu = 20

f_min_all_sorts = f - lambd*AT_Af_min_g
# plt.figure(4)
# plt.imshow(f_min_all_sorts)

def thresholdFunction(coeffs,tRange,tVal):
    bit = []
    arr,coeff_slices = pywt.coeffs_to_array(coeffs)
    for i in range (tRange):
        small_bit = coeffs[i]
        bit.append(np.shape(small_bit)[1])
    new_arr = arr
    for i in range (np.sum(bit)):
        for j in range (np.sum(bit)):
            new_arr[i][j] = pywt.threshold(np.array([arr[i][j]]),tVal,'soft')
    coeffsT = pywt.array_to_coeffs(new_arr, coeff_slices, output_format='wavedec2')
    return coeffsT

#loop through all this
def iterative(f):
    #first we do sinogram of f
    sinogram_id, Afk = astra.create_sino(f, projector_id,  returnData=True)

    #then we add noise to it to make g
    g = astra.functions.add_noise_to_sino(sinogram,100)
    # sinogram_id, g = astra.create_sino(forig, projector_id,  returnData=True)

    #take these away from each other
    Af_min_g = Afk - g

    #Set up params for AT of Af_min_g
    Af_min_g_id = astra.data2d.create('-sino',proj_geom,Af_min_g)
    cfg = astra.astra_dict('BP')
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = Af_min_g_id
    cfg['ProjectorId'] = projector_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    AT_Af_min_g = astra.data2d.get(rec_id)

    #f minus all sorts
    f_min_all_sorts = f - lambd*AT_Af_min_g

    #wavelet decomposition
    wave = pywt.wavedec2(f_min_all_sorts,'haar',level = 6)
    #threshold function
    Essed = thresholdFunction(wave,5,mu*lambd)
    #Essed = wave
    #wavelet recomposition
    inv_wave = pywt.waverec2(Essed,'haar')

    return inv_wave
    # return f_min_all_sorts

# theta = 0.01
# sigma = 2
# f = gaussian_filter(f,sigma) + theta*np.random.randn(w,h)
f = np.ones([128,128])

plt.figure(0)
plt.imshow(forig)
plt.colorbar()
print(np.sum(abs(forig - f)))

fbef = f
i=0
while np.sum(abs(forig - fbef)) >= np.sum(abs(forig - f)) and i <= 50:
# for i in range(70):
    fbef = f
    i=i+1
    f = iterative(f)
    print(np.sum(abs(forig - f)))
    # plt.figure(i)
    # plt.imshow(f)

#checking what the noisy g actually looks like
gNoisy_id = astra.data2d.create('-sino',proj_geom,sinogram+g)
rec_id = astra.data2d.create('-vol', vol_geom)
# Set up the parameters for a reconstruction via back-projection
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = gNoisy_id
cfg['ProjectorId'] = projector_id
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run back-projection and get the reconstruction
astra.algorithm.run(alg_id)
f_rec = astra.data2d.get(rec_id)
plt.figure(1)
plt.imshow(f_rec)
plt.colorbar()


print(np.sum(abs(forig - f_rec)))
plt.figure(2)
plt.imshow(f)
plt.colorbar()

plt.show()