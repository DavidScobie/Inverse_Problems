import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

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
plt.plot(W1,label='Singular values')
plt.legend()
#c  
ind = np.linspace(1,n,n)                        
mean = 0                  
sigma = sum(W1*(ind-mean)**2)/n 
# print(mean)
print(sigma)

del_n = 1

# start = del_n/((np.sqrt(2*np.pi))*sigma)

end=[]
g=[]
for i in range (n):
    end.append(np.exp(-((ind[i]-mean)**2)/(2*((sigma)**2))))
    g.append(end[i])
 
plt.plot(g,label='fit')  
plt.legend()

print(float(sigma)**2)
# print(np.var(W1))

#d
con=np.matmul(A,f)
plt.figure(2)
plt.plot(x,con)
plt.xlabel("x")
plt.ylabel("A*f")

#e
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
sarr[round(1.7*n):round(2.3*n),round(0.7*n):round(1.3*n)] = s[round(0.2*n):round(0.8*n),round(0.2*n):round(0.8*n)]
sarr[round(0.7*n):round(1.3*n),round(1.7*n):round(2.3*n)] = s[round(0.2*n):round(0.8*n),round(0.2*n):round(0.8*n)]

A_bound = sarr[n:2*n,n:2*n]

FA_bound = np.fft.fftshift(np.fft.fft(np.fft.fftshift(A_bound)))
Fconbou=FA_bound*Ff
con2bou = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Fconbou))))
plt.figure(4)
plt.plot(x,np.sum(con2bou, axis=0))

plt.show()