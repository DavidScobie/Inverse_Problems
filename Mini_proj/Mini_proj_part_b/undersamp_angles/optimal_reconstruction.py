import numpy as np
import astra
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse import identity
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

#Load phantom in
f = np.load('SLphan.npy')
plt.figure(0)
plt.imshow(f)
plt.colorbar()
forig = f

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
gtog = np.transpose(sinogram)

#Full angle with undersampled projections
new_sino = np.zeros([150,180])
for i in range (no_samples):
    new_sino[:,int(180/no_samples)*i] = gtog[:,i]

#vectorise g
g = np.reshape(gtog,(int(no_samples*det_count),1))

#mask
mask = np.zeros([150,180])
for i in range (no_samples):
    mask[:,int(180/no_samples)*i] = np.ones([150])

# Create projector geometries
no_samples1 = 180
angles = np.linspace(0,np.pi,no_samples1,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
projector_id = astra.create_projector('strip', proj_geom, vol_geom)

# Create a data object for the reconstruction
grecon_id = astra.data2d.create('-sino',proj_geom,np.transpose(new_sino))
rec_id = astra.data2d.create('-vol', vol_geom)
# Set up the parameters for a reconstruction via back-projection
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = grecon_id
cfg['ProjectorId'] = projector_id
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run back-projection and get the reconstruction
astra.algorithm.run(alg_id)
f_rec = astra.data2d.get(rec_id)
plt.figure(4)
plt.imshow(f_rec)
plt.colorbar()

flat_mask = np.reshape(mask,(150*180,1))

#Constructing the big sparse I matrix
count = -1
I = sparse.csr_matrix(np.zeros([int(no_samples)*150,1]))
for i in range (180*150):
    if flat_mask[i] == 1:
        count = count + 1
        array = np.zeros([int(no_samples)*150,1])
        array[count] = 1
        sp_arr = sparse.csr_matrix(array)
        I = sparse.hstack([I,sp_arr])
    if flat_mask[i] == 0:
        sp_arr = sparse.csr_matrix(np.zeros([int(no_samples)*150,1]))
        I = sparse.hstack([I,sp_arr])
I = sparse.lil_matrix(sparse.csr_matrix(I)[:,1:])

#Constructing IT
IT = sparse.csr_matrix.transpose(sparse.csr_matrix(I))

#Constructing laplacian
mid = np.ones([1,180]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,180,180)

D1x2d = sparse.kron(scipy.sparse.identity(180),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(180))

D2d = scipy.sparse.vstack([D1x2d,D1y2d])

D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d
Lapl = sparse.lil_matrix(sparse.csr_matrix(DT_D)[0:(180*150),0:(180*150)])

#Next implement the gmres krylov solver
alpha = 0.0067

z = lambda f: (((IT@I)-(alpha*-Lapl))*f).ravel()

A = LinearOperator((180*150,180*150),matvec = z)

ATg = lambda g: (IT@sparse.csr_matrix(g)).toarray().ravel()

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

counter = gmres_counter()

gmresOutput = gmres(A,ATg(g), x0 = np.zeros((150,180)).ravel(), callback=counter, atol=1e-06)

grecon = np.reshape(gmresOutput[0],(150,180))
plt.figure(6)
plt.imshow((grecon),cmap='gray')

#Filtered back projection

# Create a data object for the reconstruction
grecon_id = astra.data2d.create('-sino',proj_geom,np.transpose(grecon))
rec_id = astra.data2d.create('-vol', vol_geom)
# Set up the parameters for a reconstruction via back-projection
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = grecon_id
cfg['ProjectorId'] = projector_id
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run back-projection and get the reconstruction
astra.algorithm.run(alg_id)
f_rec = astra.data2d.get(rec_id)
plt.figure(7)
plt.imshow(f_rec)
plt.colorbar()

#Denoising in image domain
new_alph = 0.5
g_new = np.reshape(f_rec,(int(128**2),1))
new_I = identity(128**2)

#New Laplacian
mid = np.ones([1,128]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,128,128)

D1x2d = sparse.kron(scipy.sparse.identity(128),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(128))

D2d = scipy.sparse.vstack([D1x2d,D1y2d])

D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d
Lapl = sparse.lil_matrix(sparse.csr_matrix(DT_D)[0:(128**2),0:(128**2)])

alpha = 0.12

z = lambda f: ((new_I-(alpha*-Lapl))*f).ravel()

A = LinearOperator((128**2,128**2),matvec = z)

ATg = lambda g: (new_I@sparse.csr_matrix(g_new)).toarray().ravel()

counter = gmres_counter()

gmresOutput = gmres(A,ATg(g), x0 = np.zeros((128,128)).ravel(), callback=counter, atol=1e-06)

grecon = np.reshape(gmresOutput[0],(128,128))

plt.figure(9)
plt.imshow(grecon)
plt.colorbar()

plt.show()