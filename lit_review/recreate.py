import numpy as np
from scipy.linalg import orth
import cvxpy as cp
import matplotlib.pyplot as plt
import random

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
A1 = np.random.randn(300,1024)

samp = np.zeros(1024)
# samp[0:50]=1

pos_inds=[]
for i in range(25):
    pos_inds.append(int(np.round(1024*random.random())))
print(pos_inds)

for posi in pos_inds:
    samp[posi] = 1

neg_inds=[]
for i in range(25):
    neg_inds.append(int(np.round(1024*random.random())))
print(neg_inds)

for negi in neg_inds:
    samp[negi] = -1

# Construct the problem with no noise
f = cp.Variable(1024)

print(f)

y=A1@samp
print(y.shape)
objective = cp.Minimize(cp.sum(cp.abs(f)))
constraints = [cp.sum_squares(A1*f - y) <= 0]
prob = cp.Problem(objective, constraints)

result = prob.solve()
print(result)
print(f.value.shape)

plt.figure(0)
plt.plot(f.value)

#problem with noise =0.05
f_1 = cp.Variable(1024)
objective = cp.Minimize(cp.sum(cp.abs(f_1)))
constraints = [cp.sum_squares(A1*f_1 - y) <= 0.86]
prob = cp.Problem(objective, constraints)
result = prob.solve()
print(f_1.value)

plt.figure(1)
plt.plot(f_1.value)
plt.show()


