import numpy as np
import astra
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import sparse
from scipy.sparse import csr_matrix

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
no_samples = 30
angles = np.linspace(0,np.pi,no_samples,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
# Radon transform (generate sinogram)
SLphantom = f
sinogram_id, sinogram = astra.create_sino(SLphantom, projector_id,  returnData=True)
print(np.shape(sinogram))
transed = np.transpose(sinogram)
plt.figure(1)
plt.imshow(transed)
plt.xlabel('angle')
plt.ylabel('projection sample')
plt.colorbar()

#Full angle with undersampled projections
new_sino = np.zeros([150,180])
for i in range (no_samples):
    new_sino[:,6*i] = transed[:,i]

plt.figure(2)
plt.imshow(new_sino)
plt.xlabel('angle')
plt.ylabel('projection sample')
plt.colorbar()

# Create projector geometries
no_samples = 180
angles = np.linspace(0,np.pi,no_samples,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
# Radon transform (generate sinogram)
SLphantom = f
sinogram_id, sinogram = astra.create_sino(SLphantom, projector_id,  returnData=True)
transed = np.transpose(sinogram)

#Limited angle
new_sino1 = transed
new_sino1[:,60:120] = np.zeros([150,60])
plt.figure(3)
plt.imshow(new_sino1)
plt.xlabel('angle')
plt.ylabel('projection sample')
plt.colorbar()

#Constructing the big sparse I matrix
mask = np.ones([150,180])
mask[:,60:120] = np.zeros([150,60])
print(np.shape(mask))
flat_mask = np.reshape(mask,(150*180,1))
print(np.shape(flat_mask[0]))
# mind = np.where(mask==1)
# print(np.shape(mind))
# Isparse = scipy.sparse.identity(180*150)
# Isparse2 = scipy.sparse.identity(180*150)
# print(np.shape(mind[0]))

big = sparse.csr_matrix(np.zeros([180*150,1]))
print(np.shape(big))
for i in range (180*150):
    if flat_mask[i] == 1:
        array = np.zeros([180*150,1])
        array[i] = 1
        sp_arr = sparse.csr_matrix(array)
        big = sparse.hstack([big,sp_arr])
    if flat_mask[i] == 0:
        sp_arr = sparse.csr_matrix(np.zeros([180*150,1]))
        big = sparse.hstack([big,sp_arr])
big = sparse.lil_matrix(sparse.csr_matrix(big)[:,1:])
print(np.shape(big))

        
# A = Isparse[mind,:]

# #Filtered back projection
# # Create a data object for the reconstruction
# rec_id = astra.data2d.create('-vol', vol_geom)
# # Set up the parameters for a reconstruction via back-projection
# cfg = astra.astra_dict('FBP')
# cfg['ReconstructionDataId'] = rec_id
# cfg['ProjectionDataId'] = sinogram_id
# cfg['ProjectorId'] = projector_id
# # Create the algorithm object from the configuration structure
# alg_id = astra.algorithm.create(cfg)
# # Run back-projection and get the reconstruction
# astra.algorithm.run(alg_id)
# f_rec = astra.data2d.get(rec_id)
# plt.figure(3)
# plt.imshow(f_rec)
# plt.colorbar()
# plt.show()