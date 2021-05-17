import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

mat_data = sio.loadmat('D:\hello\ex8_movies.mat')

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
Y = mat_data['Y']

# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
R = mat_data['R']

mat_data = sio.loadmat('D:\hello\ex8_movieParams.mat')
X = mat_data['X']
print(X.shape)
theta = mat_data['Theta']
num_users = mat_data['num_users'].ravel()[0]
num_movies = mat_data['num_movies'].ravel()[0]
num_features = mat_data['num_features'].ravel()[0]


def cost(params,Y, num_users, num_movies, num_features,R,l):

    X = params[0:num_movies*num_features].reshape((num_movies, num_features))
    theta = params[num_movies*num_features: ].reshape((num_users, num_features))

    j = 0.5 * (np.sum(np.sum(R * np.square(X.dot(theta.T) - Y))))

    X_grad = (R *(X.dot(theta.T) - Y)).dot(theta)
    theta_grad = (R *(X.dot(theta.T) - Y)).T.dot(X)

    #regularize
    j = j + 0.5 * l * np.sum(np.square(theta)) + 0.5 * l * np.sum(np.square(X))

    X_grad = X_grad + l * X
    theta_grad = theta_grad + l * theta

    grad = np.hstack((X_grad.ravel(), theta_grad.ravel()))

    return j,grad


def load_movie_list():

    movie_list =[]
    with open("D:\hello\movies_list.txt") as f:
        for line in f:
            movie_list.append(line[line.index(' ') + 1:].rstrip())
    return movie_list

movie_list = load_movie_list()

my_ratings = np.zeros(len(movie_list), dtype=np.int)
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[11] = 5
my_ratings[100] = 5
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5


def normalization(Y, R):

    Y_mean = np.zeros(Y.shape[0])
    Y_new = np.zeros(Y.shape)

    for i in range(Y.shape[0]):
        Y_mean[i] = np.mean(Y[i, R[i, :] != 0])
        Y_new[i, R[i, :] != 0] = Y[i, R[i, :] != 0] - Y_mean[i]

    return Y_new, Y_mean

mat_data = sio.loadmat('D:\hello\ex8_movies.mat')

Y = mat_data['Y']
R = mat_data['R']

# Add our own ratings to the data matrix
Y = np.hstack((my_ratings.reshape(len(movie_list), 1),Y))
R = np.hstack((my_ratings.reshape(len(movie_list), 1) != 0 , R))

Y_new, Y_mean = normalization(Y, R)

# Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
theta = np.random.randn(num_users, num_features)

par = np.hstack((X.ravel(), theta.ravel()))
l = 0.0001

from scipy import optimize as opt

par, nfeval ,un = opt.fmin_tnc(func = cost,x0=par,args=(Y_new, num_users, num_movies, num_features,R,l))

X = par[0:num_movies * num_features].reshape((num_movies, num_features))
Theta = par[num_movies * num_features:].reshape((num_users, num_features))


p = X.dot(Theta.T)
result = p[:, 0] + Y_mean

rate =[]
index = []
for i in range(len(result)):
    if result[i] > 4.7:
        rate.append(result[i])
        index.append(i)

print("top rate movie recommended")
for i in range(len(rate)):
    print('rate',rate[i],movie_list[index[i]])


