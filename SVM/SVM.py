import numpy as np
from scipy.optimize import minimize
import math


"""
SVM primal with stochastic gradient descent
"""
def SVM_primal_sgd(X, y, epochs, C, gamma, schedule_func, ep=1e-06):
    # make a copy 
    X = np.array(X)
    y = np.array(y)
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
    # make a copy 
    X = np.array(X)
    y = np.array(y)

    # optimize
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
Below are whole bunch of helper functions for optimization
"""
def H_matrix(X, y):
    H = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range (X.shape[0]):
            H[i, j] = np.dot(X[i], X[j]) * y[i] * y[j]

    return H


def loss_func(alphas, H):
    return 0.5 * np.dot(alphas, np.dot(H, alphas)) - np.sum(alphas)


def constraint_func(alphas, y):
    return np.dot(alphas, y)


def jac(alphas, H):
    return np.dot(alphas.T, H) - np.ones(alphas.shape[0])

