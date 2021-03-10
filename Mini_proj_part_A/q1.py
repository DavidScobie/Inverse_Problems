import numpy as np
import astra
import matplotlib.pyplot as plt

f = np.load('SLphan.npy')
plt.figure(0)
# fplot = plt.imshow(f,cmap = 'gray')
plt.imshow(f)
plt.colorbar()

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
plt.figure(1)
plt.imshow(sinogram)
plt.colorbar()

print(sinogram_id)
plt.show()