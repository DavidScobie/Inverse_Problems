import numpy as np
import astra
import matplotlib.pyplot as plt

f=np.zeros([64,64])
f[12,12]=1
print(f)
angles = np.linspace(0,np.pi,45,endpoint=False)

#Regular form
v,h = f.shape
vol_geom = astra.create_vol_geom(v,h)
# Create projector geometries
det_count = 95
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
# Radon transform (generate sinogram)
sinogram_id, sinogram = astra.create_sino(f, projector_id,  returnData=True)
print(np.shape(sinogram))
plt.figure(0)
plt.imshow(np.transpose(sinogram))
plt.xlabel('angle')
plt.ylabel('projection sample')

A=np.zeros([45*95,64**2])
# Loop over all pixels
for i in range (64**2):
# for i in range (200):
    print(i)
    f_flat=np.zeros([1,64**2])
    f_flat[0,i] = 1
    f = np.reshape(f_flat,(64,64))
    v,h = f.shape
    vol_geom = astra.create_vol_geom(v,h)
    proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
    projector_id = astra.create_projector('strip', proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(f, projector_id,  returnData=True)
    A[:,i] = np.reshape(sinogram,(45*95,))

plt.figure(1)
plt.imshow(A)

#Check to see if result correct
f_flat_test = np.zeros(64**2,)
f_flat_test[780] = 1
g_vec = np.matmul(A,f_flat_test)
g = np.transpose(np.reshape(g_vec,(45,95)))
plt.figure(2)
plt.imshow(g)
plt.show()
