import numpy as np
import matplotlib.pyplot as plt
import cv2

#a
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

# plt.figure(0)
plt.xlabel("x")
plt.ylabel("f")
# plt.plot(x,f)


#b
n=200
del_n=2/n
std = 0.2
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
# plt.plot(W1)

print(np.var(W1))

con=np.matmul(A,f)
plt.figure(1)
plt.plot(x,con)
plt.xlabel("x")
plt.ylabel("A*f")

Ff = np.fft.fftshift(np.fft.fft(np.fft.fftshift(f)))
print(Ff.shape)
FA = np.fft.fftshift(np.fft.fft(np.fft.fftshift(A)))
print(FA.shape)
Fcon=FA*Ff
# Fcon = np.matmul(FA,Ff)
con2 = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Fcon))))
print(con2.shape)
plt.figure(2)
plt.plot(x,np.sum(con2, axis=0))

plt.show()