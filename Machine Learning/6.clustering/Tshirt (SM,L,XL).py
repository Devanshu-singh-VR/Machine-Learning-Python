import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

mat_data = sio.loadmat('D:\hello\ex7data1.mat')
X = mat_data['X']

def get_closest_points_to_each_centroid(X,centroid):

    m = X.shape[0]
    z = np.zeros(m)
    for i in range(m):
        min_cost = np.sum(np.square(centroid - X[i, :]), axis=1)
        z[i] = np.argmin(min_cost)

    return z

def mean(X,z,k):

    m,n = X.shape
    centroid = np.zeros((k,n))

    for k in range(k):
        x = X[z==k]
        centroid[k,:] = np.mean(x,axis=0)

    return(centroid)

k=3
centroid = np.array([[4,2], [6,4], [7,6]])
for i in range(11):

    z = get_closest_points_to_each_centroid(X,centroid)

    centroid = mean(X,z,k)

plt.scatter(X[z==0,0],X[z==0,1],color = 'r')
plt.scatter(X[z==1,0],X[z==1,1],color = 'b')
plt.scatter(X[z==2,0],X[z==2,1],color = 'y')
plt.show()





