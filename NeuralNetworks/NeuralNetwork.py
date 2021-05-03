import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def NeuralNetwork_sgd(X, Y, epochs, learning_rate=0.1, layers=[4], initialization='random'):
    m = X.shape[1]
    costs = []
    if initialization=='zeros':
        parameters = initialize_parameters_zeros(X, Y, layers)
    else:
        parameters = initialize_parameters_random(X, Y, layers)

    for t in range(epochs):
        # shuffle
        rand_idx = np.random.permutation(range(m))
        X = X[:, rand_idx]
        Y = Y[:, rand_idx]

        for i in range(m):
            y_hat, cache = forward_propagation(X[:, i], parameters)
            cost = compute_cost(Y[:, i], y_hat)
            grads = backward_propagation(Y[:, i], y_hat, parameters, cache)
            parameters = update_parameters(parameters, grads, learning_rate)
            costs.append(cost)

    plt.plot(range(len(costs)), costs)
    plt.show()

    return parameters


def initialize_parameters_random(X, Y, layers):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    parameters = dict()

    for i in range(len(layers)):
        n_h = layers[i]
        # first layer
        if i == 0:
            W = np.random.randn(n_h, n_x) * 0.01
            b = np.zeros((n_h, 1))
        else:
            W = np.random.randn(n_h, layers[i-1]) * 0.01
            b = np.zeros((n_h, 1))
        parameters['W' + str(i+1)] = W
        parameters['b' + str(i+1)] = b
        
    # last layer
    W = np.random.randn(n_y, layers[-1]) * 0.01
    b = np.zeros((n_y, 1))
    parameters['W' + str(len(layers)+1)] = W
    parameters['b' + str(len(layers)+1)] = b
        
    return parameters


def initialize_parameters_zeros(X, Y, layers):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    parameters = dict()

    for i in range(len(layers)):
        n_h = layers[i]
        # first layer
        if i == 0:
            W = np.zeros((n_h, n_x))
            b = np.zeros((n_h, 1))
        else:
            W = np.zeros((n_h, layers[i-1]))
            b = np.zeros((n_h, 1))
        parameters['W' + str(i+1)] = W
        parameters['b' + str(i+1)] = b
        
    # last layer
    W = np.zeros((n_y, layers[-1]))
    b = np.zeros((n_y, 1))
    parameters['W' + str(len(layers)+1)] = W
    parameters['b' + str(len(layers)+1)] = b
        
    return parameters


def forward_propagation(X, parameters):
    cache = dict()

    A = X[:, np.newaxis]
    cache['Z0'] = X[:, np.newaxis]
    cache['A0'] = X[:, np.newaxis]

    for i in range(len(parameters) // 2):
        W = parameters['W' + str(i+1)]
        Z = np.dot(W, A)
        A = sigmoid(Z)
        cache['Z' + str(i+1)] = Z
        cache['A' + str(i+1)] = A

    return A, cache

def compute_cost(Y, Y_hat):
    cost = -1 * (np.dot(Y, np.log(Y_hat).T) + np.dot((1-Y), np.log(1-Y_hat).T))
    return cost


def backward_propagation(Y, Y_hat, parameters, cache):
    grads = dict()
    dZ = Y_hat - Y
        
    for i in reversed(range(len(parameters)//2)):
        A = cache['A' + str(i)]
        dW = np.dot(dZ, A.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        grads['dW' + str(i+1)] = dW
        grads['db' + str(i+1)] = db
        W = parameters['W' + str(i+1)]
        dA_prev = np.dot(W.T, dZ)
        Z_prev = cache['Z'+str(i)]
        dZ_prev = dA_prev * sigmoid_prime(Z_prev)
        dZ = dZ_prev

    return grads


def update_parameters(parameters, grads, learning_rate):
    for i in range(len(parameters) // 2):
        W = parameters['W' + str(i+1)]
        b = parameters['b' + str(i+1)]
        dW = grads['dW' + str(i+1)]
        db = grads['db' + str(i+1)]

        W = W - learning_rate * dW
        b = b - learning_rate * db

        parameters['W' + str(i+1)] = W
        parameters['b' + str(i+1)] = b
    
    return parameters


def predict(X, parameters):
    A = X
    for i in range(len(parameters) // 2):
        W = parameters['W' + str(i+1)]
        Z = np.dot(W, A)
        A = sigmoid(Z)

    predictions = np.where(A >= 0.5, 1, 0)
    return predictions

    
def score(Y, Y_hat):
    m = Y.shape[1]
    return np.sum(Y == Y_hat) / m


def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 
        
    
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))



def main():
    Data_train = pd.read_csv('./bank-note/train.csv', header=None)
    Data_test = pd.read_csv('./bank-note/test.csv', header=None)
    X_train, Y_train = Data_train.values[:, :-1], Data_train.values[:,-1]
    X_test, Y_test = Data_test.values[:, :-1], Data_test.values[:,-1]
    Y_train = Y_train[:, np.newaxis]
    Y_test = Y_test[:, np.newaxis]
    X_train, Y_train = X_train.T, Y_train.T
    X_test, Y_test = X_test.T, Y_test.T

    params = NeuralNetwork_sgd(X_train, Y_train, epochs=10, learning_rate=0.1, layers=[25, 25], initialization='zeros')
    res_train = predict(X_train, params)
    res_test = predict(X_test, params)
    print(score(Y_train, res_train))
    print(score(Y_test, res_test))

    

if __name__=="__main__":
    main()