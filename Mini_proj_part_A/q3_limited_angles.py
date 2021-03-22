import numpy as np
import astra
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse import spdiags
from scipy import sparse
import scipy.ndimage

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
angles = np.linspace(0,0.5*np.pi,180,endpoint=False)
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


#TK0 reg
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

alpha = 0.11

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
plt.title('TK0 full range small no. angles')
plt.colorbar()



# TK1 reg
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

#Constructing gradient operator
mid = np.ones([1,128]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,128,128)

D1x2d = sparse.kron(scipy.sparse.identity(128),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(128))

print(D1x2d.shape)
# plt.figure(3)
# plt.imshow(np.reshape(D1x2d@np.reshape(f,(128**2,1)),(128,128)))
# plt.title('Gradient in x direction')

D2d = scipy.sparse.vstack([D1x2d,D1y2d])

D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d

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

alpha = 0.0015

z = lambda f: AT(A(f)) + + (alpha*(DT_D@sparse.csr_matrix(np.reshape(f,(128**2,1))).toarray())).ravel()

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

plt.figure(4)
plt.imshow(np.reshape(gmresOutput[0],(128,128)))
plt.title('TK1 full range small no. angles')
plt.colorbar()

plt.show()