import math as mt
import matplotlib.pyplot as plt
import numpy as np
a=-1
b=[]
for i in range(1,22):
    b.append(a)
    a=a+0.1
print(b)
b = np.array(b)
y=[]
for i in range(len(b)):
    y.append(mt.sqrt(-(b[i]**2) +1))
y=np.array(y)
plt.plot(b,y)
k=range(-1,1)
k = np.array(k)
p=[]
for i in range(len(k)):
    p.append(-(mt.sqrt(-(k[i]**2) +1)))
p=np.array(p)
plt.plot(k,p)
plt.show()