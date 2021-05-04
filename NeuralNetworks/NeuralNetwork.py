import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Neural network model with stochastic gradient descent
@param X: features
@param Y: labels
@param epochs: maximum iteration
@param learning_rate: initial learning rate r0 for updating parameters
@param r_func: schedule function to update learning rate based on iteration
@param layers: array representation of each hidden layer's width
@param initialization: initialization methods for parameters
@return a dictionary containing weights and biases for the network
"""
def NeuralNetwork_sgd(X, Y, epochs, learning_rate, r_func, layers=[4], initialization='random'):
    m = X.shape[1]
    costs = []
    # initialize parameters
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(X, Y, layers)
    else:
        parameters = initialize_parameters_random(X, Y, layers)

    for t in range(epochs):
        # shuffle
        rand_idx = np.random.permutation(range(m))
        X = X[:, rand_idx]
        Y = Y[:, rand_idx]

        # stochastic gradient descent
        for i in range(m):
            y_hat, cache = forward_propagation(X[:, i], parameters)
            cost = compute_cost(Y[:, i], y_hat)
            grads = backward_propagation(Y[:, i], y_hat, parameters, cache)
            parameters = update_parameters(parameters, grads, r_func(learning_rate, t))
            costs.append(cost)

    # plt.plot(range(len(costs)), costs)
    # plt.show()
    return parameters


"""
initialize parameters with random weights generated from Gaussian distribution
@param X: features
@param Y: labels
@param layers: array representation of each hidden layer's width
@param initialization: initialization methods for parameters
@return parameters being initialized
"""
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


"""
initialize parameters with zeros
@param X: features
@param Y: labels
@param layers: array representation of each hidden layer's width
@param initialization: initialization methods for parameters
@return a dictionary of parameters being initialized
"""
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


"""
forward propagation algorithm
@param X: features
@param parameters: contains weights (W) and biases (b)
@return final activation from forward propagation
@return a dictionary contains intermediate A and Z
"""
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


"""
calculate cost between predicted and actual label
@param Y: actual label
@param Y_hat: predicted label
@return cost
"""
def compute_cost(Y, Y_hat):
    cost = -1 * (np.dot(Y, np.log(Y_hat).T) + np.dot((1-Y), np.log(1-Y_hat).T))
    return cost


"""
backward propagation
@param Y: actual label
@param Y_hat: predicted label
@param parameters: contains weights (W) and biases (b)
@param cache: contains intermediate A and Z from forward propagation
@ return a dictionary of gradients
"""
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


"""
update parameters with gradients
@param parameters: contains weights (W) and biases (b)
@param grads: gradients from backward propagation
@param learning_rate: a constant for updating W and b with gradients
@return updated parameters
"""
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


"""
predict on data based on parameters
"""
def predict(X, parameters):
    A = X
    for i in range(len(parameters) // 2):
        W = parameters['W' + str(i+1)]
        Z = np.dot(W, A)
        A = sigmoid(Z)

    predictions = np.where(A >= 0.5, 1, 0)
    return predictions


"""
calculate hit rate from prediction
"""
def score(Y, Y_hat):
    m = Y.shape[1]
    return np.sum(Y == Y_hat) / m


"""
helper functions: sigmoid and sigmoid prime,
used in forward and backward propagation
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

