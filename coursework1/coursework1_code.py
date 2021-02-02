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

#######

x1 = np.linspace(-6,6,10)
x2=(5-x1)/2

plt.plot(x1,x2,'-',color='red')

plt.plot(solution[0,:], solution[1,:],'x',color='black')
plt.axis([-0.5,2,1,3])
plt.xlabel('x_1')
plt.ylabel('x_2')

#######

hermit=(np.linalg.pinv([A]))*5
print(hermit)
#The Moore Penrose Inverse corresponds to p=2.
plt.plot(x1,x2,'-',color='red')
plt.plot(solution[0,:], solution[1,:],'x',color='black')
plt.axis([-0.5,2,1,3]) 
plt.plot(hermit[0],hermit[1],'x',color='green')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.grid()
plt.show()

########

import cv2
import scipy as sp
from scipy.sparse import spdiags

n=100
x = np.linspace(-1, 1, num=n)

del_n = 2/(n-1)
std=0.2
mu=0

start = del_n/((np.sqrt(2*np.pi))*std)

end=[]
g=[]
for i in range (n):
    end.append(np.exp(-((x[i]-mu)**2)/(2*((std)**2))))
    g.append(end[i]*start)

plt.figure(0)
plt.xlabel("x")
plt.ylabel("G(x)")
plt.plot(x,g)

#######

A_big=[]
for i in range (n):
    for j in range (n):
        A_big.append(start*(np.exp(-((x[i]-x[j])**2)/(2*((std)**2)))))

#Now need to split list into list of lists to make 2D array
A = [ A_big [i:i + n] for i in range(0, len(A_big), n) ]


Atmp = np.array(np.ceil(A/np.max(A)*256), dtype = np.uint8)
Aimg = cv2.applyColorMap(Atmp, cv2.COLORMAP_JET)
plt.imshow(Aimg)
# cv2.imwrite("Aimage3.png",Aimg)

########

U, W1, VT = np.linalg.svd(A)

W = np.zeros((n, n),float)
np.fill_diagonal(W, W1)

A1=np.matmul(U,W)
A2=np.matmul(A1,VT)

normA = np.linalg.norm(A)
normB = np.linalg.norm(A2)
print(normA-normB)

######

W_dag_diag=[]
for i in range (n):
    for j in range (n):
        if i == j:
            W_dag_diag.append(1/(W[i][i]))
# print(W_dag_diag)

W_dag = np.zeros((n, n),float)
np.fill_diagonal(W_dag,W_dag_diag)
# print(W_dag)

W_Wdag = np.matmul(W,W_dag)
print(W_Wdag)
Wdag_W = np.matmul(W_dag,W)
print(np.max(W_Wdag-Wdag_W))

V=VT.transpose()
# print(V)

A_dag_1 = np.matmul(V,W_dag)
A_dag = np.matmul(A_dag_1,U.transpose())
A_Adag = np.matmul(A,A_dag)
print(np.round(A_Adag,1))
Adag_A = np.matmul(A_dag,A)
print(np.max(Adag_A-A_Adag))

#as can be seen, the check minus A dagger is approximately zero everywhere, as required
MPI = (np.linalg.pinv([A]))
print(np.max(MPI-A_dag))

#######

diagV = np.diag(V)
diagW = np.diag(W)
VP=[]
WP=[]

plt.figure(1)
plt.semilogy(diagW)
plt.xlabel("column number")
plt.ylabel("log(W)")

plt.figure(2)
fig, axs = plt.subplots(3, 3)

col=[]
for i in range (9):
    for j in range (n):
        col.append(V[j][i])
 
    ind = round(3*((i/3)-(np.floor(i/3))))
    pos1 = ind-1   
    pos0 = int(np.floor(i/3)) 
    axs[pos0, ind].plot(col)
    Column=i+1
    axs[pos0, ind].set_title("Column of V =%1.0f" %Column)   
    col=[]
for ax in axs.flat:
    ax.set(xlabel='', ylabel='V')

plt.figure(3)
fig, axs = plt.subplots(3, 3)

col=[]
for i in range (9):
    column=i+91
    for j in range (n):
        col.append(V[j][column])   
    ind = round(3*((i/3)-(np.floor(i/3))))
    pos1 = ind-1   
    pos0 = int(np.floor(i/3)) 
    axs[pos0, ind].plot(col)
    Column=column+1
    axs[pos0, ind].set_title("Column of V =%1.0f" %Column)   
    col=[]
for ax in axs.flat:
    ax.set(xlabel='', ylabel='V')

#########

#a
n=200
a=np.zeros((1,round(n/40)))
b=np.ones((1,round((7*n)/40)))
c=np.hstack((a,b))
d=0.2*np.ones((1,round((8*n)/40)))
e=np.hstack((c,d))
f=-0.5*np.ones((1,round((8*n)/40)))
g=np.hstack((e,f))
h=np.zeros((1,round((4*n)/40)))
i=np.hstack((g,h))
j=0.7*np.ones((1,round((4*n)/40)))
k=np.hstack((i,j))
l=-0.7*np.ones((1,round((8*n)/40)))
f=np.hstack((k,l)).reshape((n, 1))
x = np.linspace(-1, 1, num=n)

plt.figure(0)
plt.xlabel("x")
plt.ylabel("f")
plt.plot(x,f)

#########

#b
del_n=2/n
stds = [0.2,0.1,0.05]
for std in (stds):
    # std = 0.05
    start = del_n/(((2*np.pi)**0.5)*std)
    A_big=[]
    for i in range (n):
        for j in range (n):
            A_big.append(start*(np.exp(-((x[i]-x[j])**2)/(2*((std)**2)))))
    #Now need to split list into list of lists to make 2D array
    A = [ A_big [i:i + n] for i in range(0, len(A_big), n) ]

    Atmp = np.array(np.ceil(A/np.max(A)*256), dtype = np.uint8)
    Aimg = cv2.applyColorMap(Atmp, cv2.COLORMAP_JET)
    cv2.imwrite("Aimage3.png",Aimg)

    U, W1, VT = np.linalg.svd(A)
    plt.figure(1)
    plt.title('Singular values of A')
    plt.plot(W1,label=('std of A:',std))
    plt.legend()

########

plt.plot(W1,label=('std of A:',std))
#c  
ind = np.linspace(1,n,n)                        
mean = 0                  
sigma = sum(W1*(ind-mean)**2)/n 

del_n = 1

end=[]
g=[]
for i in range (n):
    end.append(np.exp(-((ind[i]-mean)**2)/(2*((sigma)**2))))
    g.append(end[i])
 
plt.plot(g,label='fit')  
plt.legend()

print('variance:',float(sigma)**2)

########

stds = [0.2,0.1,0.05]
for std in (stds):
    # std = 0.05
    start = del_n/(((2*np.pi)**0.5)*std)
    A_big=[]
    for i in range (n):
        for j in range (n):
            A_big.append(start*(np.exp(-((x[i]-x[j])**2)/(2*((std)**2)))))
    #Now need to split list into list of lists to make 2D array
    A = [ A_big [i:i + n] for i in range(0, len(A_big), n) ]
    #d
    con=np.matmul(A,f)
    plt.figure(2)
    plt.plot(x,con,label=('std of A:',std))
    plt.xlabel("x")
    plt.ylabel("A*f")
    plt.title('Convolution via matrix multiplication')
    plt.legend()

########

stds = [0.2,0.1,0.05]
for std in (stds):
    # std = 0.05
    start = del_n/(((2*np.pi)**0.5)*std)
    A_big=[]
    for i in range (n):
        for j in range (n):
            A_big.append(start*(np.exp(-((x[i]-x[j])**2)/(2*((std)**2)))))
    #Now need to split list into list of lists to make 2D array
    A = [ A_big [i:i + n] for i in range(0, len(A_big), n) ]
    #e
    Ff = np.fft.fftshift(np.fft.fft(np.fft.fftshift(f)))
    FA = np.fft.fftshift(np.fft.fft(np.fft.fftshift(A)))
    Fcon=FA*Ff
    con2 = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Fcon))))
    plt.figure(3)
    plt.plot(x,np.sum(con2, axis=0),label=('std of A:',std))
    plt.xlabel("x")
    plt.ylabel("A*f")
    plt.title('Convolution via fourier transform')
    plt.legend()

#########

#f
Arep = np.tile(A, (3,3))
s=np.array(A)
sarr = np.array(Arep)
sarr[round(1.7*n):round(2.3*n),round(0.7*n):round(1.3*n)] = s[round(0.2*n):round(0.8*n),round(0.2*n):round(0.8*n)]
sarr[round(0.7*n):round(1.3*n),round(1.7*n):round(2.3*n)] = s[round(0.2*n):round(0.8*n),round(0.2*n):round(0.8*n)]

A_bound = sarr[n:2*n,n:2*n]
plt.imshow(A_bound)

######

stds = [0.2,0.1,0.05]
for std in (stds):
    # std = 0.05
    start = del_n/(((2*np.pi)**0.5)*std)
    A_big=[]
    for i in range (n):
        for j in range (n):
            A_big.append(start*(np.exp(-((x[i]-x[j])**2)/(2*((std)**2)))))
    #Now need to split list into list of lists to make 2D array
    A = [ A_big [i:i + n] for i in range(0, len(A_big), n) ]
    #f
    Arep = np.tile(A, (3,3))
    s=np.array(A)
    sarr = np.array(Arep)
    sarr[round(1.7*n):round(2.3*n),round(0.7*n):round(1.3*n)] = s[round(0.2*n):round(0.8*n),round(0.2*n):round(0.8*n)]
    sarr[round(0.7*n):round(1.3*n),round(1.7*n):round(2.3*n)] = s[round(0.2*n):round(0.8*n),round(0.2*n):round(0.8*n)]

    A_bound = sarr[n:2*n,n:2*n]

    FA_bound = np.fft.fftshift(np.fft.fft(np.fft.fftshift(A_bound)))
    Fconbou=FA_bound*Ff
    con2bou = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Fconbou))))
    plt.figure(4)
    plt.plot(x,np.sum(con2bou, axis=0),label=('std of A:',std))
    plt.xlabel("x")
    plt.ylabel("A*f")
    plt.title('Convolution with boundary conditions')
    plt.legend()  