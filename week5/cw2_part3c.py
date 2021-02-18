import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse.linalg import lsqr
from scipy.sparse import spdiags
from scipy import sparse
from scipy.sparse import vstack

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

#a
img = mpimg.imread('Cameraman256.png')
f = np.float32(img)
forig=f
long_f = forig.ravel()

plt.figure(0)
imgplot = plt.imshow(f,cmap = 'gray')
plt.colorbar()


#b
sigma = 5
theta = 0.01

g = scipy.ndimage.filters.gaussian_filter(f,sigma)
w,h = g.shape
g = g + theta*np.random.randn(w,h)
plt.figure(1)
plt.imshow(g,cmap='gray')

#3

mid = np.ones([1,257]).flatten()
dat=np.array([-mid,mid])
diags_x = np.array([0,-1])
D1x = spdiags(dat,diags_x,257,256)

D1x2d = sparse.kron(scipy.sparse.identity(256),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(256))

#try to sort out proper DT_D with syntax later
# D2d = np.vstack([D1x2d,D1y2d])
# DT_D = (np.transpose(D2d))@D2d

lap = -((np.transpose(D1x2d)@D1x2d) + (np.transpose(D1y2d)@D1y2d))
A1 = scipy.sparse.identity(256**2)-(0.25*lap)


DP=[]
rsd_sq=[]
sum_Tik=[]

powers = np.linspace(-6,-3,num=13)
alphas = np.exp(powers)
for alpha in (alphas):
    diff = []
    Tik=[]
    TK1 = lambda s:(s**2)/2

    y = lambda f: gaussian_filter(f,sigma)

    z = lambda f: gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel() + np.reshape(alpha*(-lap@sparse.csr_matrix(np.reshape(f,(256**2,1))).toarray()),(1,256**2)).ravel()

    A = LinearOperator((256**2,256**2),matvec = z)

    b = np.zeros((256**2,1))

    ATg = lambda g: gaussian_filter(np.reshape(g,(256,256)),sigma).ravel()

    counter = gmres_counter()

    gmresOutput = gmres(A,ATg(g), x0 = np.zeros((256,256)).ravel(), callback=counter)
    # print(counter.niter)
    long_f = forig.ravel()
    long_gmres = gmresOutput[0].ravel()

    for i in range (256**2):
        diff.append(np.abs(long_f[i]-long_gmres[i]))
        Tik.append(TK1(long_gmres[i]))

    residual = np.sum(diff)
    sum_Tik.append(np.sum(Tik))
    rsd_sq.append(residual**2)   
    DP.append(((residual**2)/(256**2))-(theta**2))

plt.figure(2)
plt.imshow(np.reshape(gmresOutput[0],(256,256)),cmap='gray')

plt.figure(3)
plt.plot(alphas,DP)

plt.figure(4)
print(rsd_sq)
print(sum_Tik)
plt.loglog(rsd_sq,sum_Tik)


#d
DP=[]
long_lsqr=[]
rsd_sq=[]
sum_Tik=[]

D1x = spdiags(dat,diags_x,256,256)
D1x2d = sparse.kron(scipy.sparse.identity(256),D1x)
D1y2d = sparse.kron(D1x,scipy.sparse.identity(256))
D_2D = vstack([D1x2d,D1y2d])
D_2D_trans = np.transpose(D_2D)

for alpha in (alphas):
    diff = []
    Tik=[]
    TK1 = lambda s:(s**2)/2

    def M_f(f):
        top = gaussian_filter(f,sigma).ravel()
        middle = (alpha**0.5)*(D1x2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))).toarray().ravel()
        bottom = (alpha**0.5)*(D1y2d@sparse.csr_matrix(np.reshape(f,(256**2,1)))).toarray().ravel()
        return np.vstack([top,middle,bottom])

    def MT_b(b):
        b1 = np.reshape(b[0:65536],(256,256))
        Ag = gaussian_filter(b1,sigma)

        b2 = np.reshape(b[65536:],(2*(256**2),1))
        reg_bit = (alpha**0.5)*np.reshape((D_2D_trans@sparse.csr_matrix(b2)),(256,256)).toarray()
        return (Ag + reg_bit).ravel()
        

    A = LinearOperator(((256**2)*3,256**2),matvec = M_f, rmatvec = MT_b)
    siz = g.size
    b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz*2,1))])
    lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1.0e-3,iter_lim = 100)

    long_lsqr = lsqrOutput[0].ravel()

    for i in range (256**2):
        diff.append(np.abs(long_f[i]-long_lsqr[i]))
        Tik.append(TK1(long_lsqr[i]))

    residual = np.sum(diff)   
    sum_Tik.append(np.sum(Tik))
    rsd_sq.append(residual**2)  
    DP.append(((residual**2)/(256**2))-(theta**2)) 

print(lsqrOutput[2])
print(lsqrOutput[3])
print(lsqrOutput[8])

plt.figure(5)
plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

# print(np.reshape(lsqrOutput[0],(256,256))-np.reshape(gmresOutput[0],(256,256)))

plt.figure(6)
plt.plot(alphas,DP)

plt.figure(7)
plt.loglog(rsd_sq,sum_Tik)
print(rsd_sq)
print(sum_Tik)
plt.show()



