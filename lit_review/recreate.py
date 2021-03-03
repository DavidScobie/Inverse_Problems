import numpy as np
from scipy.linalg import orth
import cvxpy as cp

#Constructing A
A1 = np.random.randn(300,1024)
U = orth(A1)
V = orth(np.transpose(A1))

print(U.shape)
print(V.shape)

W_size = 300
W_diag = np.linspace(1,W_size,W_size)
W=np.zeros([W_size,W_size])
for i in range (W_size):
    for j in range (W_size):
        if i == j:
            W[i][j] = W_diag[i]

U_W = np.matmul(U,W)

A = np.matmul(U_W,np.transpose(V))
print(A.shape)

y = np.random.randn(300)

# Construct the problem.
f = cp.Variable(1024)
objective = cp.Minimize(cp.sum(f))
constraints = [cp.sum_squares(A*f - y) <= 0]
prob = cp.Problem(objective, constraints)

# print(cp.sum_squares(A*f - y))
result = prob.solve()
print(f.value)

# print(f)
