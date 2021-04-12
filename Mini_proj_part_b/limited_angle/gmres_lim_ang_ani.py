import numpy as np
import astra
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import dia_matrix

#Load phantom in
f = np.load('SLphan.npy')
plt.figure(0)
plt.imshow(f)
plt.colorbar()

# #Radon transform
# # Create volume geometries
v,h = f.shape
vol_geom = astra.create_vol_geom(v,h)

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

plt.figure(0)
plt.imshow(transed)

#Limited angle
new_sino1 = transed
new_sino1[:,60:120] = np.zeros([150,60])
plt.figure(3)
plt.imshow(new_sino1)
plt.xlabel('angle')
plt.ylabel('projection sample')
plt.colorbar()

#Filtered back projection of sinogram with missing window
# Create a data object for the reconstruction
window_sino_id = astra.data2d.create('-sino',proj_geom,np.transpose(new_sino1))
rec_id = astra.data2d.create('-vol', vol_geom)
# Set up the parameters for a reconstruction via back-projection
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = window_sino_id
cfg['ProjectorId'] = projector_id
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run back-projection and get the reconstruction
astra.algorithm.run(alg_id)
f_rec = astra.data2d.get(rec_id)
plt.figure(4)
plt.imshow(f_rec)
plt.colorbar()

#g
gstart = new_sino1[:,0:60]
gend = new_sino1[:,120:]
gtog = np.hstack((gstart,gend))
plt.figure(2)
plt.imshow(gtog)
g = np.reshape(gtog,(18000,1))
print(np.shape(g))

#mask
mask = np.ones([150,180])
mask[:,60:120] = np.zeros([150,60])
print(np.shape(mask))
plt.figure(1)
plt.imshow(mask)
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
mid = np.ones([1,180]).flatten()
dat=np.array([mid,-mid])
diags_x = np.array([0,1])
D1x = spdiags(dat,diags_x,180,180)

# plt.figure(9)
# plt.imshow(D1x.toarray())

D1x2d = sparse.kron(scipy.sparse.identity(180),D1x)
D1x2d = sparse.lil_matrix(sparse.csr_matrix(D1x2d)[0:(180*150),0:(180*150)])
D1y2d = sparse.kron(D1x,scipy.sparse.identity(180))
D1y2d = sparse.lil_matrix(sparse.csr_matrix(D1y2d)[0:(180*150),0:(180*150)])
print(np.shape(D1x2d))
print(np.shape(D1y2d))

D2d = scipy.sparse.vstack([D1x2d,D1y2d])

D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d
Lapl = sparse.lil_matrix(sparse.csr_matrix(DT_D))

#Finding threshold T
def printBoundary(a, m, n): 
    bound=[]
    for i in range(m): 
        for j in range(n): 
            if (i == 0): 
                bound.append(a[i][j])
            elif (i == m-1): 
                bound.append(a[i][j]) 
            elif (j == 0): 
                bound.append(a[i][j])
            elif (j == n-1):  
                bound.append(a[i][j])
    return bound

bound = printBoundary(gtog, 150, 120)
plt.figure(7)
plt.hist(bound,bins=30)
perc = np.percentile(bound, 70, axis=0, keepdims=True) # any number below 65 for pecentile gives error as division by zero
print(perc)

#Remaking f
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
plt.figure(1)
plt.imshow(transed)

#Finding gamma function
# T=float(perc)
T=4
del_X_f = D1x2d@sparse.csr_matrix(np.reshape(transed,(180*150,1)))
del_Y_f = D1y2d@sparse.csr_matrix(np.reshape(transed,(180*150,1)))

del_X_f_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_X_f),scipy.sparse.csr_matrix(del_X_f))
del_Y_f_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_Y_f),scipy.sparse.csr_matrix(del_Y_f))
sqrt_bit = scipy.sparse.csr_matrix.sqrt(del_X_f_squ + del_Y_f_squ)
exponent = -sqrt_bit/T
gamma_diag = (np.exp(exponent.todense()))

#plotting gamma values
plt.figure(8)
plt.imshow(np.reshape(gamma_diag,(150,180)),cmap='gray')
plt.colorbar()

gamma_diag_array = np.ravel((gamma_diag.T).sum(axis=0))
gamma = dia_matrix((gamma_diag_array, np.array([0])), shape=(180*150, 180*150))

sqrt_gam = scipy.sparse.csr_matrix.sqrt(gamma)
sqrt_gam_dx = sqrt_gam@D1x2d
sqrt_gam_dy = sqrt_gam@D1y2d
sqrt_gam_D = scipy.sparse.vstack([sqrt_gam_dx,sqrt_gam_dy])
sqrt_gam_D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(sqrt_gam_D))

plt.figure(9)
plt.imshow(sparse.lil_matrix(sparse.csr_matrix(gamma)[3100:5600,3100:5600]).toarray())

#Next implement the gmres krylov solver
alpha = 0.1

z = lambda f: (((IT@I)+(alpha*(sqrt_gam_D_trans@sqrt_gam_D)))*f).ravel()
# z = lambda f: (((IT@I)+(alpha*DT_D))*f).ravel()

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
plt.figure(5)
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
plt.figure(6)
plt.imshow(f_rec)
plt.colorbar()
plt.show()
