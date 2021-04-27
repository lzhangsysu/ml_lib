import numpy as np
import math

def logistic_reg_MAP(Data, T, r0, r_func, var):
    return logistic_reg_sgd(Data, T, r0, r_func, _gradient, var, 0.00001)

def logistic_reg_MLE(Data, T, r0, r_func, var):
    return logistic_reg_sgd(Data, T, r0, r_func, _gradient2, var, 0.00001)

def logistic_reg_sgd(Data, T, r0, r_func, grad_func, var, epsilon):
    Data = np.array(Data)
    w = np.zeros(Data.shape[1], dtype='float64')
    prev_grad = w

    for t in range(T):
        np.random.shuffle(Data)
        X_data = np.append(Data[:,:-1], np.ones((Data.shape[0], 1)), axis=1)
        y_data = Data[:, -1]

        for i in range(y_data.shape[0]):
            X = X_data[i]
            y = y_data[i]

            grad = grad_func(X, y, w, var)

            w = w - (r_func(r0, t) * grad)
            w /= np.linalg.norm(w)

            if np.linalg.norm(prev_grad - grad) < epsilon * t:
                print('converged on iter', t)
                print(w)
                return w

            prev_grad = grad
    
    return w

def logistic_reg_test(Data, w):
    err = 0.0

    for row in Data:
        X = np.append(row[:-1], 1)
        y = row[-1]

        if y == 0:
            y = -1

        if (y * np.dot(w, X)) <= 0:
            err += 1

    return err/Data.shape[0]

def sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))


def gradient_MAP(X, y, w, var):
    sig = sigmoid(np.dot(X, w))
    grad = (1.0 / var * w) + np.dot(X.T, sig-y)
    return grad

def gradient_MLE(X, y, w, var):
    sig = sigmoid(np.dot(X, w))
    grad = np.dot(X.T, sig-y) / var
    return grad


def _gradient(X, y, w, v):
    top = -X * y
    bottom = 1 + np.exp(y * w.T @ X)
    right = w / (2 * v)
    return top / bottom + right


def _gradient2(X, y, w, v):
    top = -X * y * v
    bottom = 1 + np.exp(y * w.T @ X)
    return top / bottom
