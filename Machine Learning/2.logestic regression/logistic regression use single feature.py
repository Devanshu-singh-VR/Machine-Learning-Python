import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("D:\hello\ex2.txt", "r"), delimiter=",")
x1=data[:, 0]
y=data[:, 2]

# Building the t1odel
t1 = 0
t2 = 0

alpha = 0.0001  # The learning Rate
itterations = 1000  # The nut1ber of iterations to perfort1 gradient dest2ent

n = float(len(x1)) # Nut1ber of elet1ents in X

# Perfort1ing Gradient Dest2ent
for i in range(itterations):
    ht = 1 / (1+ np.exp(-(t1*x1 + t2)))
    D_t1 = (-2/n) * sum(x1 * (y - ht) * ht * (1-ht))  # Derivative wrt t1
    D_t2 = (-2/n) * sum((y - ht) * ht * (1-ht))
    m = t1 - alpha * D_t1  # Update t1
    o = t2 - alpha * D_t2  # Update t2

    t1 = m
    t2 = o



plt.scatter(x1,y,color='r')
plt.scatter(x1,ht,color='b')
plt.show()
