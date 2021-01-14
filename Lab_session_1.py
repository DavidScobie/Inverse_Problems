import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

p=np.array([1,1.5,2,2.5,3,3.5,4])
solution=np.zeros([2,int(len(p))])

A=np.array([1,2])
b=5

def Phi(x):
    x[1] = 5 - x[0]/2
    phi=np.sum(np.abs(x[:])**p[i])
    return phi

for i in range (len(p)):
    # Phi = lambda x: np.sum(np.abs(x[:])**p[i])
    z = scipy.optimize.minimize(Phi,np.array([0, 0]), method='Nelder-Mead',tol=1e-5)
    solution[:,i]=np.asarray(z.x)
    print(solution[:,i])

x1 = np.linspace(-6,6,10)
x2=5-x1/2

plt.plot(x1,x2,'-',color='red')

plt.plot(solution[0,:], solution[1,:],'x',color='black')
plt.axis([-6,6,-6,6])
plt.grid()

plt.show()

# def phi(x,p):
#     val = sum(np.abs(x)**p)
#     return val

# x=np.array([1,1.2])
# print(phi(x,2))



# cons = ({'type': 'eq', 'fun': phi, A, b: A@x - b, 'args' : (A,b)})


