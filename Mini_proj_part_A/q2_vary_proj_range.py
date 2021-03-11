import numpy as np
import astra
import matplotlib.pyplot as plt

f_siz = 64
det_count = 95
n_projections = 45
k=-1
#Change projection range
proj_range_array=np.array([(np.pi)/4,(np.pi)/2,np.pi])
for proj_range in proj_range_array:
    A=np.zeros([n_projections*det_count,f_siz**2])
    angles = np.linspace(0,proj_range,n_projections,endpoint=False)
    for i in range (f_siz**2):
    # for i in range (200):
        print(i)
        f_flat=np.zeros([1,f_siz**2])
        f_flat[0,i] = 1
        f = np.reshape(f_flat,(f_siz,f_siz))
        v,h = f.shape
        vol_geom = astra.create_vol_geom(v,h)
        proj_geom = astra.create_proj_geom('parallel',1.,det_count,angles)
        projector_id = astra.create_projector('strip', proj_geom, vol_geom)
        sinogram_id, sinogram = astra.create_sino(f, projector_id,  returnData=True)
        A[:,i] = np.reshape(sinogram,(n_projections*det_count,))

    #Check to see if result correct
    f_flat_test = np.zeros(f_siz**2,)
    f_flat_test[195] = 1
    g_vec = np.matmul(A,f_flat_test)
    g = np.transpose(np.reshape(g_vec,(n_projections,det_count)))

    k=k+1
    plt.figure(2*k)
    plt.imshow(g)
    plt.xlabel('Projection number')
    plt.ylabel('Projection sample')

    #SVD of A
    U, W1, VT = np.linalg.svd(A)
    W = np.zeros((f_siz**2, f_siz**2),float)
    np.fill_diagonal(W, W1)

    plt.figure((2*k)+1)
    plt.title("singular values of A. Projection range in radians: %1.3f" % proj_range)  
    plt.plot(W1)

plt.show()