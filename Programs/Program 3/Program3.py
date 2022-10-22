from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
import numpy as np
import math
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    # Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J


def gradientDescent(X, y, theta, alpha, numIterations):
    m = y.size
    xTrans = X.transpose()
    J_history = zeros(shape=(numIterations, 1))

    for i in range(0, numIterations):
        # compute prediction and error
        prediction = np.dot(X, theta)
        print(prediction)
        error = prediction - y
        # compute gradient
        errors = zeros(shape=(y.size, 4))
        errors[:, 0] = y
        errors[:, 1] = prediction
        errors[:, 2] = -error
        errors[:, 3] = error * error
        if(i < 2):

            print("Errors: ")
            print(errors)
            print()

            deltas = np.tile(np.array([error]).T, np.shape(X)[1])
            deltas = np.multiply(abs(deltas), X)

            print("Error Deltas: ")
            print(deltas)
            print()

        gradient = np.dot(xTrans, error)
        # update theta
        theta = theta - alpha * gradient
        if(i < 2):
            print(f'New weights after {i+1} iterations')
            print(theta)
            print()
        if (i == 99):
            print(f'Final weights after {i + 1} iterations')
            print(theta)
            print()
            print("Final square error sum")
            print(np.sum(error*error)/(2*np.shape(X)[0]))
            print()


        # compute cost of updated theta
        cost = compute_cost(X, y, theta)
        J_history[i, 0] = cost

    return theta, J_history

# part 1
print('File: prog3.txt')
data = loadtxt('prog3.txt', delimiter=',', usecols=(1,2,3,5))

X = data[:, :3]
y = data[:, 3]

X_new = ones(shape=(y.size, 4))
X_new[:, 1:4] = X
X = X_new
theta = [-0.146, 0.185, -0.044, 0.119]
alpha = 0.00000002

theta, J_history = gradientDescent(X, y, theta, alpha, 100)

plot(arange(100), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

# part 2
print('File: prog3_2.txt')
data = loadtxt('prog3_2.txt', delimiter=',')

y = data[:, 1]
X = data[:, 1:]
X_new = ones(shape=(y.size, 3))
X_new[:, 1:3] = X[:, 1:3]
X = X_new

theta = [-59.5, -0.15, 0.6]
alpha = 0.000002

theta, J_history = gradientDescent(X, y, theta, alpha, 100)

plot(arange(100), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

