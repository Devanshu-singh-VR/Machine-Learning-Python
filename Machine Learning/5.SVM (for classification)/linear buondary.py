import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

# Load from ex6data1
mat_data = sio.loadmat('D:\hello\ex6data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
X = np.array(X)

# Plot training data
y1 = np.argwhere(y==1)
y0 = np.argwhere(y==0)

plt.plot(X[y1, 0],X[y1, 1],linestyle='',marker='+',color='r')
plt.plot(X[y0, 0],X[y0, 1],linestyle='',marker='o',color='b')
plt.xlabel("X1")
plt.ylabel("X2")

# Change the C value below and see how the decision boundary varies (e.g., try C = 1000).
clf = svm.SVC(kernel = 'linear', C = 1)
clf.fit(X,y)

w = clf.coef_[0]
a = -w[0] / w[1]

xx = np.linspace(-0,5)

yy = (a * xx) - clf.intercept_[0] / w[1]

plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()

a = float(input("x1 = "))
b = float(input("x2 = "))
print('output = ',clf.predict(np.array([[a,b]])))
b = float(input("enter to exit"))

