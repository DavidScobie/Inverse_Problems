import numpy as np
from scipy.linalg import orth
import cvxpy as cp

#Constructing A
A1 = np.random.randn(300,1024)
# U = orth(A1)
# V = orth(np.transpose(A1))

samp = np.zeros(1024)
samp[0:50]=1

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

# Construct the problem.
f = cp.Variable(1024)

print(f)

y=A1@samp
print(y.shape)
objective = cp.Minimize(cp.sum(cp.abs(f)))
constraints = [cp.sum_squares(A1*f - y) <= 0.19]
prob = cp.Problem(objective, constraints)

# print(cp.sum_squares(A*f - y))
result = prob.solve()
print(result)
print(f.value.shape)

# print(f)
