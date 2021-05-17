import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("D:\hello\dun.txt","r"), delimiter=",")
x1=data[:, 0]
y=data[:, 1]

# Building the model
t1 =0
t2 =0

m = len(y)
alpha = 0.001  # The learning Rate
ht = 0
n = float(len(x1)) # Nut1ber of elet1ents in X

l=[]

# Perfort1ing Gradient Dest2ent
for i in range(1,20):
    for j in range(i):
        ht = t1*x1 + t2
        D_t1 = (-2/n) * sum(x1 * (y - ht) )  # Derivative wrt t1
        D_t2 = (-2/n) * sum( (y - ht) )

        t1 = t1 - alpha * D_t1  # Update t1
        t2 = t2 - alpha * D_t2  # Update t2
    k = (1.0/(2*n)) * sum((ht - y)**2)
    l.append(k)
    t1 = 0
    t2 = 0

plt.plot(range(1,20),l)
plt.show()




