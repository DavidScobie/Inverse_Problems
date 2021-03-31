import numpy as np
import astra
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

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

#g
gstart = new_sino1[:,0:60]
gend = new_sino1[:,120:]
gtog = np.hstack((gstart,gend))
# g = np.transpose(gtog.ravel())
g = np.reshape(gtog,(18000,1))
print(np.shape(g))

#mask
mask = np.ones([150,180])
mask[:,60:120] = np.zeros([150,60])
print(np.shape(mask))
flat_mask = np.reshape(mask,(150*180,1))

#Constructing the big sparse I matrix
count = -1
I = sparse.csr_matrix(np.zeros([120*150,1]))
print(np.shape(I))
for i in range (180*150):
    if flat_mask[i] == 1:
        count = count + 1
        array = np.zeros([120*150,1])
        array[count] = 1
        sp_arr = sparse.csr_matrix(array)
        I = sparse.hstack([I,sp_arr])
    if flat_mask[i] == 0:
        sp_arr = sparse.csr_matrix(np.zeros([120*150,1]))
        I = sparse.hstack([I,sp_arr])
I = sparse.lil_matrix(sparse.csr_matrix(I)[:,1:])
print(np.shape(I))

#Constructing IT
IT = sparse.csr_matrix.transpose(sparse.csr_matrix(I))
print(np.shape(IT))
#Constructing laplacian
mid = np.ones([1,256]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,256,256)

D1x2d = sparse.kron(scipy.sparse.identity(256),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(256))

D2d = scipy.sparse.vstack([D1x2d,D1y2d])

D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d
Lapl = sparse.lil_matrix(sparse.csr_matrix(DT_D)[0:(180*150),0:(180*150)])

#check IT*g
ITg_array = (IT@sparse.csr_matrix(g)).toarray().ravel()
print(np.shape(ITg_array.ravel()))

#Next implement the gmres krylov solver
alpha = 0.1

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

plt.figure(3)
plt.imshow(np.reshape(gmresOutput[0],(150,180)),cmap='gray')

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
plt.show()