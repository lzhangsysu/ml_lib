import numpy as np
from scipy.optimize import minimize

def SVM_sgd(Data, C, T, r0, r_rate_func, ep=1e-06):
    # initialize weight to 0, make a copy of data
    Data = np.array(Data)
    w = np.zeros(Data.shape[1], dtype='float64')
    prev_grad = w

    # iterate each epoch
    for t in range(0, T):
        np.random.shuffle(Data)

        X_data = np.append(Data[:,:-1], np.ones((Data.shape[0], 1)), axis=1) # append b=1 to X
        y_data = Data[:,-1]
        N = y_data.shape[0]

        # iterate each example
        for i in range(y_data.shape[0]):
            X = X_data[i]
            y = y_data[i]

            if y == 0: # convert label
                y = -1

            # gradient descent
            grad = w
            if max(0, 1 - y * np.dot(w, X)) != 0:
                grad = grad - C * N * y * X
                
            # update weights
            r_t = r_rate_func(r0, t)
            w = w - r_t * grad

            # check for convergence
            if np.linalg.norm(prev_grad - grad) < t * ep:
                print('converged on iter', t, 'w: ', w)
                return w

            prev_grad = grad

    return w



def SVM_test(Data, w):
    err = 0.0

    for row in Data:
        X = np.append(row[:-1], 1)
        y = row[-1]

        if y == 0:
            y = -1

        if (y * np.dot(w, X)) <= 0:
            err += 1

    return err/Data.shape[0]
