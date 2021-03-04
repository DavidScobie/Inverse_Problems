import numpy as np
from scipy.linalg import orth
import cvxpy as cp
import matplotlib.pyplot as plt
import random
import scipy as sp

# U = orth(A1)
# V = orth(np.transpose(A1))

# W_size = 300
# W_diag = np.linspace(1,W_size,W_size)
# W=np.zeros([W_size,W_size])
# for i in range (W_size):
#     for j in range (W_size):
#         if i == j:
#             W[i][j] = W_diag[i]

# U_W = np.matmul(U,W)

# A = np.matmul(U_W,np.transpose(V))
# print(A.shape)


#Constructing A
A = np.random.randn(300,1024)

#Find singular values of matrix A (as described in paper)
U, W1, VT = np.linalg.svd(A)
W = np.zeros((300, 300),float)
np.fill_diagonal(W, W1)

plt.figure(0)
plt.title('singular values of A')
plt.plot(W1)

samp = np.zeros(1024)

pos_inds=[]
for i in range(25):
    pos_inds.append(int(np.round(1024*random.random())))

for posi in pos_inds:
    samp[posi] = 1

neg_inds=[]
for i in range(25):
    neg_inds.append(int(np.round(1024*random.random())))

for negi in neg_inds:
    samp[negi] = -1

plt.figure(1)
plt.title('Sampled points')
plt.plot(samp)

# Construct the problem with no noise
f = cp.Variable(1024)

y=A@samp
objective = cp.Minimize(cp.sum(cp.norm(f,1)))
# constraints = [cp.sum_squares(A*f - y) <= 0]
constraints = [cp.norm(A@f - y,2) <= 0]
prob = cp.Problem(objective, constraints)

result = prob.solve()

plt.figure(2)
plt.title('Signal with no noise')
plt.plot(f.value)

#problem with noise = 0.05
f_1 = cp.Variable(1024)
# objective = cp.Minimize(cp.sum(cp.abs(f_1)))
objective = cp.Minimize(cp.sum(cp.norm(f_1,1)))
# constraints = [cp.sum_squares(A*f_1 - y) <= 9.34]
constraints = [cp.norm(A@f_1 - y,2) <= 9.34]
prob = cp.Problem(objective, constraints)
result = prob.solve()
print(f_1.value)

plt.figure(3)
plt.title('Signal with noise')
plt.plot(f_1.value)

#Constructing A with exponentially decreasing singular spectrum
start = np.random.randn(300,1024)
U = orth(start)
V = orth(np.transpose(start))
W_diag = np.zeros(300)
for k in range (300):
    W_diag[k] = np.exp(-(k/100))
    # W_diag[k] = 50 - (0.125*k)

W_mat = np.zeros([300,300])
np.fill_diagonal(W_mat, W_diag)

A_begin = np.matmul(U,W_mat)
A = np.matmul(A_begin,np.transpose(V))

#Finding sparse signal with exp decreasing singular values of A
f_2 = cp.Variable(1024)
objective = cp.Minimize(cp.sum(cp.norm(f_2,1)))
constraints = [cp.norm(A@f_2 - y,2) <= 0]
prob = cp.Problem(objective, constraints)
result = prob.solve()

plt.figure(4)
plt.title('Signal from A with exponentially decaying singular values')
plt.plot(f_2.value)

plt.show()


