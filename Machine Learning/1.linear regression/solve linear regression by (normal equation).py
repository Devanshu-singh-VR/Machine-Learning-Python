import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("D:\hello\dun.txt","r"), delimiter=",")
p=data[:, 0]
y=data[:, 1]

k=[]
for i in range(len(p)):
    k.append([1,p[i]])

x = np.array(k)
y = np.array(y)

t = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)


l = t[0] + t[1]*x

plt.plot(p,l,color='y')
plt.scatter(p,y,color='r')
plt.show()