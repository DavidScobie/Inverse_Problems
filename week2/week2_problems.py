import numpy as np
import matplotlib.pyplot as plt
import cv2

n=4
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

A=[]
for i in range (n):
    for j in range (n):
        A.append(start*(np.exp(-((x[i]-x[j])**2)/(2*((std)**2)))))
print(A)
#Now need to split list into list of lists to make 2D array

# Atmp = np.array(np.ceil(A/np.max(A)*256), dtype = np.uint8)
# Aimg = cv2.applyColorMap(Atmp, cv2.COLORMAP_JET)
# cv2.imwrite("Aimage3.png",Aimg)

plt.show()

