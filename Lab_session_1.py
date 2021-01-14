import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

p=np.array([1,1.5,2,2.5,3,3.5,4])
solution=np.zeros([2,int(len(p))])

A=np.array([1,2])
b=5

def Phi(x):
    #this is a rearrangement of our condition
    x[0] = 5 - 2*(x[1])
    #this is the equation we are trying to minimise
    phi=np.sum(np.abs(x[:])**p[i])
    return phi

for i in range (len(p)):
    #We initially put point (0,0) in to Phi(), and then iterate to find minimum
    z = scipy.optimize.minimize(Phi,np.array([0, 0]), method='Nelder-Mead',tol=1e-5)
    #z has many fields, we only want the x field
    solution[:,i]=np.asarray(z.x)
    print(solution[:,i])

x1 = np.linspace(-6,6,10)
x2=(5-x1)/2

plt.plot(x1,x2,'-',color='red')

plt.plot(solution[0,:], solution[1,:],'x',color='black')
plt.axis([-0.5,2,1,3])

hermit=(np.linalg.pinv([A]))*5
print(hermit)
#The Moore Penrose Inverse corresponds to p=2. 
plt.plot(hermit[0],hermit[1],'x',color='green')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.grid()
plt.show()



