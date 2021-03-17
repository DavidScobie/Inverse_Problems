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
theta = 0.01
g = f
w,h = g.shape
g = g + theta*np.random.randn(w,h)
plt.figure(1)
plt.imshow(g)
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
# print(np.shape(sinogram))
plt.figure(1)
plt.imshow(np.transpose(sinogram))
plt.xlabel('angle')
plt.ylabel('projection sample')
plt.colorbar()

projector_id = astra.create_projector('strip', proj_geom, vol_geom)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ProjectorId'] = projector_id
alg_id = astra.algorithm.create(cfg)

print(projector_id)
print(rec_id)
print(alg_id)

# def A(f,projector_id):
def A(f):
    print(f.shape)
    print(projector_id)
    f_resh = np.reshape(f,(128,128))
    # sinogram_id, sinogram = astra.create_sino(f_resh, projector_id,  returnData=True)
    sinogram_id, sinogram = astra.create_sino(f_resh, 5,  returnData=True)
    A_x = sinogram
    return A_x

# def AT(y,rec_id,alg_id):
def AT(y):
    print(y.shape)
    # astra.algorithm.run(alg_id)
    astra.algorithm.run(7)
    # f_rec = astra.data2d.get(rec_id)
    f_rec = astra.data2d.get(6)
    AT_y = f_rec
    print(rec_id)
    print(alg_id)
    print(AT_y.shape)
    return AT_y.ravel()

alpha = 0.1

# y = AT(A(f,projector_id),rec_id,alg_id) + alpha*f

# z = lambda f: AT(A(f,5),6,7) + 0.1*f
z = lambda f: AT(A(f)) + 0.1*f

A1 = LinearOperator((128**2,128**2),matvec = z)

# ATg = lambda g: AT(g,rec_id,alg_id).ravel()
ATg = lambda g: AT(g).ravel()

gmresOutput = gmres(A1,ATg(g), x0 = np.zeros((128,128)).ravel(), atol=1e-06)

plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(128,128)),cmap='gray')



# def M_f(f):
#     v,h = f.shape
#     det_count = 150
#     angles = np.linspace(0,np.pi,180,endpoint=False)
#     proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
#     vol_geom = astra.create_vol_geom(v,h)
#     projector_id = astra.create_projector('strip', proj_geom, vol_geom)
#     top = astra.create_sino(f, projector_id,  returnData=True)

#     top = gaussian_filter(f,sigma).ravel()
#     bottom = (alpha**0.5)*np.reshape(f,(256,256)).ravel()
#     return np.vstack([top,bottom])

# def MT_b(b):
#     b1 = np.reshape(b[0:128**2],(128,128))
#     b2 = np.reshape(b[128**2:],(128,128))

#     v,h = b1.shape
#     vol_geom = astra.create_vol_geom(v,h)
#     rec_id = astra.data2d.create('-vol', vol_geom)
#     cfg = astra.astra_dict('FBP')
#     cfg['ReconstructionDataId'] = rec_id
#     cfg['ProjectionDataId'] = sinogram_id
#     cfg['ProjectorId'] = projector_id

#     return (gaussian_filter(b1,sigma) + np.sqrt(alpha)*b2).ravel()
    

# A = LinearOperator(((128**2)*2,128**2),matvec = M_f, rmatvec = MT_b)
# siz = g.size
# b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
# lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1e-06)
# print(lsqrOutput[2])

# plt.figure(3)
# plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

plt.show()