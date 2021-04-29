import numpy as np
import math

def logistic_reg_MAP(X, y, T, r0, r_func, var):
    return logistic_reg_sgd(X, y, T, r0, r_func, var, 'MAP')


def logistic_reg_MLE(X, y, T, r0, r_func, var):
    return logistic_reg_sgd(X, y, T, r0, r_func, var, 'MLE')


def logistic_reg_sgd(X, y, T, r0, r_func, var, model='MAP'):
    N = X.shape[0]
    w = np.zeros(X.shape[1], dtype='float64')

    for t in range(T):
        rand_idx = np.random.permutation(N)
        X = X[rand_idx]
        y = y[rand_idx]

        for i in range(N):
            X_data = X[i]
            y_data = y[i]

            if model == 'MAP':
                grad = grad_MAP(X_data, y_data, w, var)
            else:
                grad = grad_MLE(X_data, y_data, w, var)
            w = w - (r_func(r0, t) * grad)

    return w


def logistic_reg_test(X, y, w):
    err = 0.0

    for i in range(X.shape[0]):
        X_data = X[i]
        y_data = y[i]

        if y_data == 0:
            y_data = -1

        if (y_data * np.dot(w, X_data)) <= 0:
            err += 1

    return err/X.shape[0]


def grad_MAP(X, y, w, v):
    top = -X * y * v
    sigmoid = 1 + np.exp(y * w.T @ X)
    prior = w / (2 * v)
    return top / sigmoid + prior


def grad_MLE(X, y, w, v):
    top = -X * y * v
    sigmoid = 1 + np.exp(y * w.T @ X)
    return top / sigmoid


