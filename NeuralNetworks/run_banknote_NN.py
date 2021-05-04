import numpy as np
import pandas as pd
import NeuralNetwork


# read data
def load_data(file_path):
    Data = pd.read_csv(file_path, header=None)
    X, Y = Data.values[:, :-1], Data.values[:,-1]
    Y = Y[:, np.newaxis]
    X, Y = X.T, Y.T

    return X, Y

X_train, Y_train = load_data("./bank-note/train.csv")
X_test, Y_test = load_data("./bank-note/test.csv")


# schedule function for learning rate
def r_func(r0, t):
    d = 1
    r = r0 / (1 + (r0/d)*t)
    return r


# hidden layer widths
widths = [5, 10, 25, 50, 100]


# 3 layer neural network with random initialization
print('random initialization')
print('w & err_train & err_test')
for w in widths:
    params = NeuralNetwork.NeuralNetwork_sgd(
        X_train, Y_train, epochs=20, 
        learning_rate=0.1, r_func=r_func, 
        layers=[w, w])
    pred_train = NeuralNetwork.predict(X_train, params)
    pred_test = NeuralNetwork.predict(X_test, params)
    err_train = 1 - NeuralNetwork.score(Y_train, pred_train)
    err_test = 1 - NeuralNetwork.score(Y_test, pred_test)
    print(w, '&', err_train, '&', err_test)


# 3 layer neural network with zeros initialization
print('\ninitialize with zeros')
print('w & err_train & err_test')
for w in widths:
    params = NeuralNetwork.NeuralNetwork_sgd(
        X_train, Y_train, epochs=20, 
        learning_rate=0.1, r_func=r_func, 
        layers=[w, w], initialization='zeros')
    pred_train = NeuralNetwork.predict(X_train, params)
    pred_test = NeuralNetwork.predict(X_test, params)
    err_train = 1 - NeuralNetwork.score(Y_train, pred_train)
    err_test = 1 - NeuralNetwork.score(Y_test, pred_test)
    print(w, '&', err_train, '&', err_test)
