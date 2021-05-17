#import libraries
import numpy as np
import scipy as sp

#import data
x = np.array(([3,5],[5,1],[10,2]) , dtype='float')
y = np.array(([0.66],[0.85],[0.94]) , dtype='float')

#New complete class, with changes:
class Neural_Network:
    def __init__(self, Lambda=0):
        #Define Hyperparameters
        self.inputs = 2
        self.outputs = 1
        self.hidden = 3

        #Weights (parameters)
        self.w1 = np.random.randn(self.inputs,self.hidden)
        self.w2 = np.random.randn(self.hidden,self.outputs)

        #Regularization Parameter:
        self.Lambda = Lambda

    def forward(self, x):
        #Propogate inputs though network
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        ht = self.sigmoid(self.z3)
        return ht

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoid_d(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, x, y):
        #Compute cost for given x,y, use weights already stored in class.
        self.ht = self.forward(x)
        J = 0.5*sum((y-self.ht)**2)/x.shape[0] + (self.Lambda/2)*(np.sum(self.w1**2)+np.sum(self.w2**2))
        return J

    def gradient(self, x, y):
        #Compute derivative with respect to W and w2 for a given x and y:
        self.ht = self.forward(x)

        delta3 = np.multiply(-(y-self.ht), self.sigmoid_d(self.z3))
        dJdw2 = np.dot(self.a2.T, delta3)/x.shape[0] + self.Lambda*self.w2

        delta2 = np.dot(delta3, self.w2.T)*self.sigmoid_d(self.z2)
        dJdw1 = np.dot(x.T, delta2)/x.shape[0] + self.Lambda*self.w1

        return np.hstack((dJdw1.ravel(), dJdw2.ravel()))

    #Helper functions for interacting with other methods/classes
    def vector_pram(self):
        #Get w1 and w2 Rolled into vector:
        pram = np.hstack((self.w1.ravel(), self.w2.ravel()))
        return pram

    def reshaped_pram(self, pram):
        #Set w1 and w2 using single parameter vector:
        self.w1 = np.reshape(pram[0:self.hidden*self.inputs],(self.inputs, self.hidden))
        self.w2 = np.reshape(pram[self.hidden*self.inputs:],(self.hidden, self.outputs))


from scipy import optimize as opt

##Need to modify trainer class a bit to check testing error during training:
class trainer:
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def cost(self, pram, x, y):
        self.N.reshaped_pram(pram)
        cost = self.N.costFunction(x, y)
        grad = self.N.gradient(x,y)

        return(cost, grad)

    def train(self,x,y):

        pram0 = self.N.vector_pram()

        pram, nfeval ,un = opt.fmin_tnc(func = self.cost,x0=pram0,args=(x,y))

        self.N.reshaped_pram(pram)
        self.optimizationResults = pram


NN = Neural_Network(Lambda=0.00001)
T = trainer(NN)
T.train(x,y)

print(NN.forward(x))

