from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel


sigmoid = lambda x: 1 / (1 + np.exp(-x))

def print_data(y, errors, deltas):
    print('Target'.ljust(10), 'Pred'.ljust(15), 'Error'.ljust(15), 'Error Sqr'.ljust(15),
          'delta(w[0])'.ljust(15), 'delta(w[1])'.ljust(15), 'delta(w[2])'.ljust(15))
    for i in range(5):
        print(str(y[i]).ljust(10), str(round(errors[i][1], 8)).ljust(15), str(round(errors[i][2], 8)).ljust(15),
              str(round(errors[i][3], 8)).ljust(15), str(round(deltas[i][0], 8)).ljust(15),
              str(round(deltas[i][2], 8)).ljust(15), str(round(deltas[i][0], 8)).ljust(15))
    print(f'Sum of Square Errors = {round(np.sum(errors[:, 3])/2, 4)}\n')



def plot_data(X, y, theta, n):
    positive = []
    negative = []
    for i in range(y.size):
        if y[i]:
            positive.append(X[i])
        else:
            negative.append(X[i])

    positive = np.array(positive)
    negative = np.array(negative)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(positive[:, 1], positive[:, 2], s=50, c='b', marker='o')
    ax.scatter(negative[:, 1], negative[:, 2], s=50, c='r', marker='x')
    x = np.linspace(-1, 1, 1000)
    ax.plot(x, -(theta[0]/theta[2]) - (theta[1]/theta[2]) * x,
            label=f'{round(-theta[0]/theta[2], 2)}+{round(-theta[1]/theta[2], 2)}r iter={n}')
    ax.legend()
    ax.set_xlabel('RPM')
    ax.set_ylabel('Vibration')


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    # Number of training samples
    m = y.size
    predictions = X.dot(theta)
    predictions = sigmoid(predictions)
    sqErrors = (predictions - y)

    J = (1.0 / 2) * sqErrors.T.dot(sqErrors)


    return (1.0 / m) * J


def gradientDescent(X, y, theta, alpha, numIterations):
    m = y.size
    xTrans = X.transpose()
    J_history = zeros(shape=(numIterations, 1))

    for i in range(0, numIterations):
        # compute prediction and error
        prediction = np.dot(X, np.array(theta))
        prediction = sigmoid(prediction)
        error = prediction - y
        # Calculate errors and deltas
        errors = zeros(shape=(y.size, 4))
        errors[:, 0] = y
        errors[:, 1] = prediction
        errors[:, 2] = -error
        errors[:, 3] = error * error
        deltas = np.tile(np.array([-error]).T, np.shape(X)[1])
        deltas = np.multiply(deltas, X)
        preds = np.tile(np.array([prediction]).T, np.shape(X)[1])

        deltas = np.multiply(deltas, preds)
        deltas = np.multiply(deltas, 1 - preds)

        # Display Prediciton, Error, and Deltas
        if(i < 2 or i == 1999):
            print_data(y, errors, deltas)
            print((np.sum(deltas, axis=0)))

        # Plot Decision Boundary
        if i+1 in [1, 10, 200, 500, 2000]:
            plot_data(X, y, theta, i+1)

        theta = theta + (alpha * np.sum(deltas, axis=0))


        # Display weights
        if(i < 2 or i == 1999):
            print(f'New weights after {i+1} iterations')
            print(theta)
            print()




        cost = compute_cost(X, y, theta)
        J_history[i, 0] = cost


    return theta, J_history


print('File: Table7_7.txt')
data = loadtxt('Table7_7.txt', delimiter=',')

X = data[:, :3]
y = data[:, -1]
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(X)

X_new = ones(shape=(y.size, 3))
X_new[:, 1:4] = X[:, 1:3]
X = X_new

print("First 5 rows of normalized data:")
print(X[:5, 1:])
print()

theta = [-2.9465, -1.0147, 2.161]
alpha = 0.02

print(f'Initial wieghts are:\n{theta}\n')

theta, J_history = gradientDescent(X, y, theta, alpha, 2000)

# Cost Plot
ax = plt.subplots(figsize=(8,8))
plot(arange(2000), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()



