import numpy as np
import matplotlib.pyplot as plt
import cv2

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

# B = [[1,2,3],[4,5,6]]
# print(B)

# Atmp = np.array(np.ceil(B/np.max(B)*256), dtype = np.uint8)
# Aimg = cv2.applyColorMap(Atmp, cv2.COLORMAP_JET)
# cv2.imwrite("Aimage3.png",Aimg)

plt.show()

