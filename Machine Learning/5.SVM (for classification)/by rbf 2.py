import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

# Load from ex6data1
mat_data = sio.loadmat('D:\hello\ex6data2.mat')
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
plt.show()
# Change the C value below and see how the decision boundary varies (e.g., try C = 1000).
clf = svm.SVC(kernel = 'rbf',gamma='scale',C=1000)
clf.fit(X,y)

#accuracy
print('accuracy : ',clf.score(X,y))

k = np.linspace(0,1)
h = []
for i in range(len(k)):
    for j in range(len(k)):
        if clf.predict(np.array([[k[i],k[j]]])) == 0:
            h.append([k[i],k[j]])

h = np.array(h)
plt.plot(X[y1, 0],X[y1, 1],linestyle='',marker='+',color='r')
plt.plot(X[y0, 0],X[y0, 1],linestyle='',marker='o',color='b')
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(h[:,0],h[:,1],color='g')
plt.title("zero area bound by green")
plt.show()

a = float(input("x1 = "))
b = float(input("x2 = "))
print('output = ',clf.predict(np.array([[a,b]])))
b = float(input("enter to exit"))