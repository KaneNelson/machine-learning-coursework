import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot(X, y, theta):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.scatter(X[:25, 0], X[:25, 1], s=50, c='b', marker='o')
    ax.scatter(X[25:50, 0], X[25:50, 1], s=50, c='r', marker='+')
    ax.scatter(X[50:, 0], X[50:, 1], s=50, c='black', marker='x')

    x = np.linspace(-1, 1, 1000)
    colors = ['b', 'r', 'black']
    for i in range(3):
        ax.plot(x, -(theta[i][0]/theta[i][2]) - (theta[i][1]/theta[i][2]) * x, c=colors[i],
                label=f'{round(-theta[i][0]/theta[i][2], 2)}+{round(-theta[i][1]/theta[i][2], 2)}s')
    ax.legend()

    ax.set_xlabel('SPEND')
    ax.set_ylabel('FREQ')
    plt.show()

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    return np.sum(first - second) / (len(X)) + reg


def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    # grad[0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    grad[0] = grad[0] - learningRate / len(X) * theta[0]

    return np.array(grad).ravel()


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x
    print('Final Theta values')
    print(all_theta)



    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

def predict_oneinstance(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # Predictions
    h = sigmoid(X * all_theta.T)
    print('Predictions are:')
    print('Model 1          Model 2         Model 3')
    print(np.array(h)[0])
    print()

    #Normalized Predictions
    h = h/np.sum(h)
    print('Normalized predictions are: ')
    print('Model 1          Model 2         Model 3')
    print(np.array(h)[0])

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    print('\nPredicted model class is: ')
    print(np.array(h_argmax)[0][0])
    return h_argmax


def main():
    # load the dataset from mat format file
    data = np.loadtxt('Table7_11.txt', delimiter=',')
    X = data[:, :2]
    scaler = MinMaxScaler((-1, 1))
    X = scaler.fit_transform(X)
    y = data[:, -1]



    print(X.shape, y.shape)
    rows = X.shape[0]
    params = X.shape[1]

    # make a matrix 10X(params+1), each row stores a vector of thetas
    num_labels = 3
    all_theta = np.zeros((num_labels, params + 1))

    # generate all theta values using one-vs-all classification
    all_theta = one_vs_all(X, y, num_labels, 0.0001)

    plot(X, y, all_theta)

    # predict all original dataset using the trained models
    y_pred = predict_all(X, all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%\n'.format(accuracy * 100))

    predict_oneinstance(np.array([[0.10790978, 0.7643608]]), all_theta)


# invoke the main
main()
