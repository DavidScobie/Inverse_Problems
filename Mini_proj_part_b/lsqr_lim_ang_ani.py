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
SLPha = np.load('SLphan.npy')

# #Radon transform
# # Create volume geometries
v,h = SLPha.shape
vol_geom = astra.create_vol_geom(v,h)

# Create projector geometries
no_samples = 180
angles = np.linspace(0,np.pi,no_samples,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
# Radon transform (generate sinogram)
sinogram_id, sinogram = astra.create_sino(SLPha, projector_id,  returnData=True)
f = np.transpose(sinogram)

#Limited angle
new_sino1 = f
new_sino1[:,60:120] = np.zeros([150,60])

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

#g
gstart = new_sino1[:,0:60]
gend = new_sino1[:,120:]
gtog = np.hstack((gstart,gend))
g = np.reshape(gtog,(18000,1))

#mask
mask = np.ones([150,180])
mask[:,60:120] = np.zeros([150,60])
flat_mask = np.reshape(mask,(150*180,1))

#Constructing the big sparse I matrix
count = -1
I = sparse.csr_matrix(np.zeros([120*150,1]))
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

#Constructing IT
IT = sparse.csr_matrix.transpose(sparse.csr_matrix(I))

#Constructing laplacian
mid = np.ones([1,180]).flatten()
dat=np.array([mid,-mid])
diags_x = np.array([0,1])
D1x = spdiags(dat,diags_x,180,180)

D1x2d = sparse.kron(scipy.sparse.identity(180),D1x)
D1x2d = sparse.lil_matrix(sparse.csr_matrix(D1x2d)[0:(180*150),0:(180*150)])
D1y2d = sparse.kron(D1x,scipy.sparse.identity(180))
D1y2d = sparse.lil_matrix(sparse.csr_matrix(D1y2d)[0:(180*150),0:(180*150)])

D2d = scipy.sparse.vstack([D1x2d,D1y2d])

D_2D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(D2d))
DT_D = D_2D_trans@D2d
Lapl = sparse.lil_matrix(sparse.csr_matrix(DT_D)[0:(180*150),0:(180*150)])

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
perc = np.percentile(bound, 70, axis=0, keepdims=True)

#Remaking f
no_samples = 180
angles = np.linspace(0,np.pi,no_samples,endpoint=False)
det_count = 150
proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(SLPha, projector_id,  returnData=True)
f = np.transpose(sinogram)

#Finding gamma function
T=float(perc)
del_X_f = D1x2d@sparse.csr_matrix(np.reshape(f,(180*150,1)))
del_Y_f = D1y2d@sparse.csr_matrix(np.reshape(f,(180*150,1)))

del_X_f_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_X_f),scipy.sparse.csr_matrix(del_X_f))
del_Y_f_squ = scipy.sparse.csr_matrix.multiply(scipy.sparse.csr_matrix(del_Y_f),scipy.sparse.csr_matrix(del_Y_f))
sqrt_bit = scipy.sparse.csr_matrix.sqrt(del_X_f_squ + del_Y_f_squ)
exponent = -sqrt_bit/T
gamma_diag = (np.exp(exponent.todense()))

gamma_diag_array = np.ravel((gamma_diag.T).sum(axis=0))
gamma = dia_matrix((gamma_diag_array, np.array([0])), shape=(180*150, 180*150))
#other gamma related matrices
sqrt_gam = scipy.sparse.csr_matrix.sqrt(gamma)
sqrt_gam_dx = sqrt_gam@D1x2d
sqrt_gam_dy = sqrt_gam@D1y2d
sqrt_gam_D = scipy.sparse.vstack([sqrt_gam_dx,sqrt_gam_dy])
sqrt_gam_D_trans = sparse.csr_matrix.transpose(scipy.sparse.csr_matrix(sqrt_gam_D))

#lsqr solver
alpha = 0.1

def M_f(f):
    # end_of_top = np.zeros([9000,1]).ravel()
    top = I@sparse.csr_matrix(np.reshape(f,(180*150,1))).toarray().ravel() #I*f
    # full_top = np.hstack[(top,end_of_top)]
    middle = (alpha**0.5)*(D1x2d@sparse.csr_matrix(np.reshape(f,(180*150,1)))).toarray().ravel() #sqrt(alpha)*Dx*f
    bottom = (alpha**0.5)*(D1y2d@sparse.csr_matrix(np.reshape(f,(180*150,1)))).toarray().ravel() #sqrt(alpha)*Dy*f
    return np.vstack([top,middle,bottom])

def MT_b(b):
    b1 = np.reshape(b[0:18000],(18000,1))
    ITg = IT@sparse.csr_matrix(np.reshape(b1,(18000,1))).ravel()

    b2 = np.reshape(b[18000:],(2*(27000),1))
    reg_bit = (alpha**0.5)*(D_2D_trans@sparse.csr_matrix(b2)).toarray().ravel()
    return (ITg + reg_bit).ravel()
    

A = LinearOperator(((27000)*3,27000),matvec = M_f, rmatvec = MT_b)
# siz = g.size
b = np.vstack([np.reshape(g,(18000,1)),np.zeros((27000*2,1))])
lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((150,180)).ravel(),atol=1.0e-6,iter_lim = 100)

plt.figure(0)
plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')
plt.show()