import numpy as np
import astra
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres

#Load phantom in
f = np.load('SLphan.npy')
plt.figure(0)
plt.imshow(f)
plt.colorbar()

#Add noise to phantom
# theta = 0.01
theta = 0.1
w,h = f.shape
g = f + theta*np.random.randn(w,h)
plt.figure(1)
plt.imshow(g)
plt.colorbar()

#Radon transform
# Create volume geometries
v,h = f.shape
vol_geom = astra.create_vol_geom(v,h)
# Create projector geometries
# angles = np.linspace(0,np.pi,180,endpoint=False)
angles = np.linspace(0,np.pi,180,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
# Radon transform (generate sinogram)
SLphantom = f
sinogram_id, sinogram = astra.create_sino(SLphantom, projector_id,  returnData=True)


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
plt.figure(2)
plt.imshow(f_rec)
plt.title('FBP reconstruction')
plt.colorbar()


# Redefining for the krylov solver
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(SLphantom, projector_id,  returnData=True)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = projector_id
alg_id = astra.algorithm.create(cfg)

print(projector_id)
print(rec_id)
print(alg_id)

def A(f):
    f_resh = np.reshape(f,(128,128))
    sinogram_id, sinogram = astra.create_sino(f_resh, projector_id,  returnData=True)
    A_x = sinogram
    return A_x

def AT(y):
    astra.algorithm.run(alg_id)
    f_rec = astra.data2d.get(rec_id)
    AT_y = f_rec
    return AT_y.ravel()

alpha = 0.1

z = lambda f: AT(A(f)) + alpha*f

A1 = LinearOperator((128**2,128**2),matvec = z)

ATg = lambda g: AT(g).ravel()

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

counter = gmres_counter()

gmresOutput = gmres(A1,ATg(g), x0 = np.zeros((128,128)).ravel(),callback=counter, atol=1e-06)

plt.figure(3)
plt.imshow(np.reshape(gmresOutput[0],(128,128)))
plt.title('output of gmres')
plt.colorbar()

plt.show()