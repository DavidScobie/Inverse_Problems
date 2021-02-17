import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse.linalg import lsqr


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        # if self._disp:
        #     print('iter %3i\trk = %s' % (self.niter, str(rk)))

#a
img = mpimg.imread('Cameraman256.png')
f = np.float32(img)
forig=f
plt.figure(0)
imgplot = plt.imshow(f,cmap = 'gray')
plt.colorbar()


#b
sigma = 2
theta = 0.01

g = scipy.ndimage.filters.gaussian_filter(f,sigma)
w,h = g.shape
g = g + theta*np.random.randn(w,h)
plt.figure(1)
plt.imshow(g,cmap='gray')


#c
alpha = 0.01
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
    z = lambda f: gaussian_filter(y(np.reshape(f,(256,256))),sigma).ravel() + alpha*f
    A = LinearOperator((256**2,256**2),matvec = z)
    ATg = lambda g: gaussian_filter(np.reshape(g,(256,256)),sigma).ravel()

    counter = gmres_counter()

    gmresOutput = gmres(A,ATg(g), x0 = f.ravel(), callback=counter)
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
plt.loglog(rsd_sq,sum_Tik)



#d
DP=[]
long_lsqr=[]

for alpha in (alphas):
    diff = []

    def M_f(f):
        top = gaussian_filter(f,sigma).ravel()
        bottom = (alpha**0.5)*np.reshape(f,(256,256)).ravel()
        return np.vstack([top,bottom])

    def MT_b(b):
        b1 = np.reshape(b[0:65536],(256,256))
        b2 = np.reshape(b[65536:],(256,256))
        return (gaussian_filter(b1,sigma) + np.sqrt(alpha)*b2).ravel()
        

    A = LinearOperator(((256**2)*2,256**2),matvec = M_f, rmatvec = MT_b)
    siz = g.size
    b = np.vstack([np.reshape(g,(siz,1)),np.zeros((siz,1))])
    lsqrOutput = scipy.sparse.linalg.lsqr(A,b, x0 = np.zeros((256,256)).ravel(),atol=1.0e-9)
    print(lsqrOutput[2])

    long_lsqr = lsqrOutput[0].ravel()

    for i in range (256**2):
        diff.append(np.abs(long_f[i]-long_lsqr[i]))

    residual = np.sum(diff)    
    DP.append(((residual**2)/(256**2))-(theta**2))    

plt.figure(5)
plt.imshow(np.reshape(lsqrOutput[0],(256,256)),cmap='gray')

plt.figure(6)
plt.plot(alphas,DP)
plt.show()


