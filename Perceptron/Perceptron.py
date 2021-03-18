import numpy as np 


"""
standard Perceptron algorith
@param Data: input data
@param T: max number of epoch
@r: learning rate
@return: weight vector
"""
def Perceptron(Data, T, r):
    # initialize weights to 0 and make a copy of data
    Data = np.array(Data)
    w = np.zeros(Data.shape[1], dtype='float64')

    for epoch in range(T):
        np.random.shuffle(Data)

        X_data = np.append(Data[:,:-1], np.ones((Data.shape[0], 1)), axis=1) # append b=1 to X
        y_data = Data[:,-1]

        for i in range(y_data.shape[0]):
            X = X_data[i]
            y = y_data[i]

            if y == 0: # convert label
                y = -1

            # update weight if weight vector misclassifies label
            if (y * np.dot(w, X)) <= 0:
                w = w + (r * y * X)

    return w


"""
voted Perceptron algorith
@param Data: input data
@param T: max number of epoch
@r: learning rate
@return: weight vectors and vote vector
"""
def Perceptron_voted(Data, T, r):
    # initialize weights and votes, and make a copy of data
    Data = np.array(Data)
    w = np.zeros(Data.shape[1], dtype='float64')
    weights_list = []
    weights_list.append(w)

    m = 0
    C = []
    C.append(1)

    for epoch in range(T):
        np.random.shuffle(Data)

        X_data = np.append(Data[:,:-1], np.ones((Data.shape[0], 1)), axis=1) # append b=1 to X
        y_data = Data[:,-1]

        for i in range(y_data.shape[0]):
            X = X_data[i]
            y = y_data[i]

            if y == 0: # convert label
                y = -1

            # update weight and increment m if weight vector misclassifies label
            if (y * np.dot(w, X)) <= 0:
                w = w + (r * y * X)
                weights_list.append(w)
                m += 1
                C.append(1)
            else:
                C[m] += 1

    return weights_list, C


"""
standard Perceptron algorith
@param Data: input data
@param T: max number of epoch
@r: learning rate
@return: average weight vector
"""
def Perceptron_average(Data, T, r):
    Data = np.array(Data)
    w = np.zeros(Data.shape[1], dtype='float64')
    a = np.zeros(Data.shape[1], dtype='float64')

    for epoch in range(T):
        np.random.shuffle(Data)

        X_data = np.append(Data[:,:-1], np.ones((Data.shape[0], 1)), axis=1) # append b=1 to X
        y_data = Data[:,-1]

        for i in range(y_data.shape[0]):
            X = X_data[i]
            y = y_data[i]

            if y == 0: # convert label
                y = -1

            if (y * np.dot(w, X)) <= 0:
                w = w + (r * y * X)

            a = a + w

    return a


"""
return error rate on test data with given weight vector
"""
def Perceptron_test(Data, w):
    error = 0.0

    for row in Data:
        X = np.append(row[:-1], 1)
        y = row[-1]

        if y == 0:
            y = -1

        if (y * np.dot(w, X)) <= 0:
            error += 1

    return error/Data.shape[0]


"""
return error rate on test data with list of weight vector and votes
"""
def Perceptron_voted_test(Data, W, C):
    error = 0.0

    for row in Data:
        X = np.append(row[:-1], 1)
        y = row[-1]

        if y == 0:
            y = -1

        s = 0
        for w_c in zip(W, C):
            w = w_c[0]
            c = w_c[1]

            # prediction = sgn(sum of c*sgn(wTx))
            pred = np.dot(w, X)
            if pred <= 0:
                s -= c
            else:
                s += c

        if (y * s) <= 0:
            error += 1

    return error/Data.shape[0]

            














