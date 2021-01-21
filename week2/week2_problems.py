import numpy as np
import matplotlib.pyplot as plt
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

A_big=[]
for i in range (n):
    for j in range (n):
        A_big.append(start*(np.exp(-((x[i]-x[j])**2)/(2*((std)**2)))))

#Now need to split list into list of lists to make 2D array
A = [ A_big [i:i + n] for i in range(0, len(A_big), n) ]


Atmp = np.array(np.ceil(A/np.max(A)*256), dtype = np.uint8)
Aimg = cv2.applyColorMap(Atmp, cv2.COLORMAP_JET)
cv2.imwrite("Aimage3.png",Aimg)


# Atmp = np.array(np.ceil(B/np.max(B)*256), dtype = np.uint8)
# Aimg = cv2.applyColorMap(Atmp, cv2.COLORMAP_JET)
# cv2.imwrite("Aimage3.png",Aimg)

U, W1, VT = np.linalg.svd(A)

W = np.zeros((n, n),float)
np.fill_diagonal(W, W1)

# print(A)
# print(U)
# print(W)
# print(VT)

A1=np.matmul(U,W)
A2=np.matmul(A1,VT)
# print(A2)

normA = np.linalg.norm(A)
normB = np.linalg.norm(A2)
# print(normA)
# print(normB)

# W_dag = np.full((n, n), 1/W[0][0])
# print(W_dag)

# spdiags(W_dag, np.array([0, -1, 2,1]), n, n).toarray()
# print(W_dag)

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
# print(W_Wdag)

V=VT.transpose()
# print(V)

A_dag_1 = np.matmul(V,W_dag)
A_dag = np.matmul(A_dag_1,U.transpose())
# print(A_dag)

#as can be seen, the check minus A dagger is approximately zero everywhere, as required
check = (np.linalg.pinv([A]))
# print(np.max(check-A_dag))

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


plt.show()

