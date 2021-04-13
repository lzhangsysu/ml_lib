import numpy as np
from scipy.optimize import minimize
import math


"""
SVM primal with stochastic gradient descent
"""
def SVM_primal_sgd(X, y, epochs, C, gamma, schedule_func, ep=1e-06):
    # add bias term
    bias = np.ones((X.shape[0], 1), dtype='float64') 
    X = np.hstack((X, bias))

    # initialize weight, gamma_0
    w = np.zeros(X.shape[1], dtype='float64')
    gamma_0 = gamma
    N = y.shape[0]
    t = 0
    grads = []

    # iterate each epoch
    for epoch in range(0, epochs):
        # randomly shuffle data
        idx = np.arange(N)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # iterate each example
        for i in range(N):
            t += 1

            # updated weights based on prediction
            y_pred = y[i] * np.dot(w, X[i])

            if y_pred <= 1:
                w = (1 - gamma) * w + gamma * C * N * y[i] * X[i]
            else:
                w = (1 - gamma) * w

            # update gamma
            gamma = schedule_func(gamma_0, t)

            # check for convergence
            grad = (1/2) * np.dot(w, w) + C * max(0, 1 - y[i]*np.dot(w, X[i]))
            if len(grads) > 0 and abs(grad - grads[-1]) < epoch * ep:
                print('converged on iter', epoch)
                return w
            grads.append(grad)

    return w


"""
SVM dual
"""
def SVM_dual(X, y, C):
    # set up parameters for optimization
    x0 = np.random.rand(X.shape[0])
    H = H_matrix(X, y)
    bounds = [(0, C)] * X.shape[0]

    # optimize
    res = minimize(loss_func, x0, args=(H,), method='L-BFGS-B', jac=jac, bounds=bounds)

    # recover w, b
    w = np.sum([res.x[i] * y[i] * X[i,:] for i in range(X.shape[0])], axis=0)
    b = np.mean(y - np.dot(X, w))

    return w, b


"""
SVM kernal
"""
def SVM_kernel(X, y, epochs, C, gamma):
    # set up parameters for optimization
    x0 = np.random.rand(X.shape[0])
    H = H_matrix_kernal(X, y, gamma, Gaussian_kernel)
    bounds = [(0, C)] * X.shape[0]

    # optimize
    res = minimize(loss_func, x0, args=(H,), method='L-BFGS-B', jac=jac, bounds=bounds)

    # recover w, b, alphas
    alphas = res.x
    w = np.sum([alphas[i] * y[i] * X[i,:] for i in range(X.shape[0])], axis=0)
    K = K_matrix(X, X, gamma, Gaussian_kernel)
    K = K * alphas * y
    b = np.mean(y - np.sum(K, axis=0))

    return w, b, alphas


"""
Prediction error of SVM primal
"""
def SVM_primal_test(X, y, w):
    # add bias term
    bias = np.ones((X.shape[0], 1), dtype='float64') 
    X = np.hstack((X, bias))
    
    err = 0.0
    for i in range(X.shape[0]):
        if np.sign(w.dot(X[i])) != y[i]:
            err += 1

    return err/X.shape[0]


"""
Prediction error of SVM dual
"""
def SVM_dual_test(X, y, w, b):
    err = 0.0

    for i in range(X.shape[0]):
        if np.sign(w.dot(X[i]) + b) != y[i]:
            err += 1

    return err/X.shape[0]


"""
Prediction error of SVM kernal
"""
def SVM_kernel_test(X, y, X_train, y_train, alphas, b, gamma):
    err = 0.0

    K = K_matrix(X_train, X, gamma, Gaussian_kernel)
    alphas = np.reshape(alphas, (alphas.shape[0], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    
    K = alphas * y_train * K
    pred = np.sum(K, axis=0) + b

    for i in range(X.shape[0]):
        if np.sign(pred[i]) != y[i]:
            err += 1

    return err/X.shape[0]


"""
Below are whole bunch of helper functions for optimization
"""
def H_matrix(X, y):
    H = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range (X.shape[0]):
            H[i, j] = np.dot(X[i], X[j]) * y[i] * y[j]

    return H


def H_matrix_kernal(X, y, gamma, kernel_func):
    H = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range (X.shape[0]):
            H[i, j] = kernel_func(X[i], X[j], gamma) * y[i] * y[j]

    return H


def K_matrix(x1, x2, gamma, kernel_func):
    K = np.zeros((x1.shape[0], x2.shape[0]))

    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            K[i, j] = kernel_func(x1[i], x2[j], gamma)

    return K


def Gaussian_kernel(x1, x2, gamma):
    return math.exp((-1) * np.linalg.norm(x1 - x2, 2) / gamma)


def loss_func(alphas, H):
    return 0.5 * np.dot(alphas, np.dot(H, alphas)) - np.sum(alphas)


def jac(alphas, H):
    return np.dot(alphas.T, H) - np.ones(alphas.shape[0])

