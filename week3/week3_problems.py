import numpy as np
import matplotlib.pyplot as plt

a=np.zeros((1,5))
b=np.ones((1,35))
c=np.hstack((a,b))
d=0.2*np.ones((1,40))
e=np.hstack((c,d))
f=-0.5*np.ones((1,40))
g=np.hstack((e,f))
h=np.zeros((1,20))
i=np.hstack((g,h))
j=0.7*np.ones((1,20))
k=np.hstack((i,j))
l=-0.7*np.ones((1,40))
f=np.hstack((k,l)).reshape((200, 1))
x = np.linspace(-1, 1, num=200)

plt.figure(0)
plt.xlabel("x")
plt.ylabel("f")
plt.plot(x,f)
plt.show()
