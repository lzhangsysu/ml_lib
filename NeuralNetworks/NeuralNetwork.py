import numpy as np


class L_layer_Neural_Network():
    def __init__(self):
        self.layers = [5]
        self.parameters = dict()
        self.cache = dict()


    def initialize_parameters(self):
        n_x = self.X.shape[0]
        n_y = self.Y.shape[0]

        for i in range(len(self.layers)):
            n_h = layers[i]
            # first layer
            if i == 0:
                W = np.random.randn(n_h, n_x) * 0.01
                b = np.zeros((n_h, 1))
            else:
                W = np.random.randn(n_h, self.layers[i-1]) * 0.01
                b = np.zeros((n_h, 1))
            self.parameters['W' + str(i+1)] = W
            self.parameters['b' + str(i+1)] = b
        # last layer
        W = np.random.randn(n_y, self.layers[-1]) * 0.01
        b = np.zeros((n_y, 1))
        self.parameters['W' + str(len(self.layers)+1)] = W
        self.parameters['b' + str(len(self.layers)+1)] = b
        
        # return parameters
        return


    def forward_propagation(self, X):
        A = X[:, np.newaxis]
        self.cache['Z0'] = X[:, np.newaxis]
        self.cache['A0'] = X[:, np.newaxis]

        for i in range(len(self.parameters) // 2):
            W = self.parameters['W' + str(i+1)]
            Z = np.dot(W, A)
            A = self.sigmoid(Z)
            self.cache['Z' + str(i+1)] = Z
            self.cache['A' + str(i+1)] = A

        return A

    
    def compute_cost(self, Y, Y_hat):
        cost = -1 * (np.dot(Y, np.log(Y_hat).T) + np.dot((1-Y), np.log(1-Y_hat).T))
        self.cost = cost
        return cost

    
    def backward_propagation(Y, Y_hat, parameters, cache):
        self.grads = dict()
        dZ = Y_hat - Y
        
        for i in reversed(range(len(parameters)//2)):
            A = cache['A' + str(i)]
            dW = np.dot(dZ, A.T)
            db = np.sum(dZ, axis=1, keepdims=True)
            self.grads['dW' + str(i+1)] = dW
            self.grads['db' + str(i+1)] = db
            W = parameters['W' + str(i+1)]
            dA_prev = np.dot(W.T, dZ)
            Z = cache['Z'+str(i+1)]
            dZ_prev = dA_prev * sigmoid_prime(Z)
            dZ = dZ_prev

        return grads

    
    def update_parameters(self, parameters, grads, learning_rate):
        for i in range(len(parameters) // 2):
            W = parameters['W' + str(i+1)]
            b = parameters['b' + str(i+1)]
            dW = grads['dW' + str(i+1)]
            db = grads['db' + str(i+1)]

            W = W - learning_rate * dW
            b = b - learning_rate * db

            parameters['W' + str(i+1)] = W
            parameters['b' + str(i+1)] = b
        return

    def predict(self, X):
        A = X
        for i in range(len(self.parameters) // 2):
            W = self.parameters['W' + str(i+1)]
            Z = np.dot(W, A)
            A = self.sigmoid(Z)

        predictions = np.where(A >= 0.5, 1, 0)
        return predictions

    def score(self, Y, Y_hat):
        m = Y.shape[1]
        return np.sum(Y == Y_hat) / m

    def sigmoid(z):
        return 1 / (1 + np.exp(-z)) 
        
    def sigmoid_prime(z):
        return sigmoid(z) * (1 - sigmoid(z))



def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 3
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def nn_model(X, Y, num_iterations = 10000):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions