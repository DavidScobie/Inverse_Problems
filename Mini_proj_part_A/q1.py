import numpy as np
import astra
import matplotlib.pyplot as plt

#Load phantom in
f = np.load('SLphan.npy')
plt.figure(0)
plt.imshow(f)
plt.colorbar()

#Radon transform
# Create volume geometries
v,h = f.shape
vol_geom = astra.create_vol_geom(v,h)
# Create projector geometries
angles = np.linspace(0,np.pi,180,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
# Radon transform (generate sinogram)
SLphantom = f
sinogram_id, sinogram = astra.create_sino(SLphantom, projector_id,  returnData=True)
print(np.shape(sinogram))
plt.figure(1)
plt.imshow(np.transpose(sinogram))
plt.xlabel('angle')
plt.ylabel('projection sample')
plt.colorbar()
non_zero_vals = np.count_nonzero(sinogram)
percent_non_zero = 100*(non_zero_vals/(np.shape(sinogram)[0]*(np.shape(sinogram)[1])))

#Unfiltered back projection
# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)
# Set up the parameters for a reconstruction via back-projection
cfg = astra.astra_dict('BP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = projector_id
# cfg['ReconstructionDataId'] = 5
# cfg['ProjectionDataId'] = 3
# cfg['ProjectorId'] = 1
print(cfg)
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run back-projection and get the reconstruction
astra.algorithm.run(alg_id)
f_rec = astra.data2d.get(rec_id)
plt.figure(2)
plt.imshow(f_rec)
plt.colorbar()

#Filtered back projection
# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)
# Set up the parameters for a reconstruction via back-projection
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = projector_id
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run back-projection and get the reconstruction
astra.algorithm.run(alg_id)
f_rec = astra.data2d.get(rec_id)
plt.figure(3)
plt.imshow(f_rec)
plt.colorbar()

#normalized reconstructed true image
im_norm = (f_rec-np.min(f_rec))/(np.max(f_rec)-np.min(f_rec))
plt.figure(4)
plt.imshow(im_norm)
plt.colorbar()

#Add noise to sinogram
gNoisy = astra.functions.add_noise_to_sino(sinogram,100)
gNoisy_id = astra.data2d.create('-sino',proj_geom,sinogram+gNoisy)
plt.figure(5)
plt.imshow(gNoisy+sinogram)
plt.colorbar()
# Create a data object for the reconstruction
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
plt.figure(6)
plt.imshow(f_rec)
plt.colorbar()

#normalized reconstructed noisy image
im_noise_norm = (f_rec-np.min(f_rec))/(np.max(f_rec)-np.min(f_rec))
plt.figure(7)
plt.imshow(im_noise_norm)
plt.colorbar()

err_squ = np.sum((im_noise_norm-im_norm)**2)

#Produce graph of error with noiselevel
err_array=[]
noise_vals = np.linspace(100,1000,num=30)
for noise in noise_vals:
    gNoisy = astra.functions.add_noise_to_sino(sinogram,noise)
    gNoisy_id = astra.data2d.create('-sino',proj_geom,sinogram+gNoisy)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = gNoisy_id
    cfg['ProjectorId'] = projector_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    f_rec = astra.data2d.get(rec_id)
    im_noise_norm = (f_rec-np.min(f_rec))/(np.max(f_rec)-np.min(f_rec))
    err_squ = (np.sum((im_noise_norm-im_norm)**2)/(128**2))**0.5
    err_array.append(err_squ)

plt.figure(8)
plt.plot(noise_vals,err_array)
plt.ylabel('RMSE')
plt.xlabel('Noise level')
plt.show()