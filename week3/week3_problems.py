import numpy as np
import matplotlib.pyplot as plt
import cv2

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


#b
del_n=2/n
std = 0.05
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
plt.plot(W1)

print(np.var(W1))

con=np.matmul(A,f)
plt.figure(2)
plt.plot(x,con)
plt.xlabel("x")
plt.ylabel("A*f")

Ff = np.fft.fftshift(np.fft.fft(np.fft.fftshift(f)))
FA = np.fft.fftshift(np.fft.fft(np.fft.fftshift(A)))
Fcon=FA*Ff
con2 = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Fcon))))
plt.figure(3)
plt.plot(x,np.sum(con2, axis=0))

#f
Arep = np.tile(A, (3,3))
s=np.array(A)
sarr = np.array(Arep)
sarr[340:460,140:260] = s[40:160,40:160]
sarr[140:260,340:460] = s[40:160,40:160]
# plt.imshow(sarr)

A_bound = sarr[200:400,200:400]
# plt.imshow(A_bound)

FA_bound = np.fft.fftshift(np.fft.fft(np.fft.fftshift(A_bound)))
# print(FA.shape)
Fconbou=FA_bound*Ff
# Fcon = np.matmul(FA,Ff)
con2bou = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Fconbou))))
# print(con2.shape)
plt.figure(3)
plt.plot(x,np.sum(con2bou, axis=0))



plt.show()