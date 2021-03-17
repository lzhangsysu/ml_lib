import numpy as np 

"""
standard Perceptron algorith
@param Data: input data
@param T: max number of epoch
@r: learning rate
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

            # check if weight vector misclassifies label
            if (y * np.dot(w, X)) <= 0:
                w = w + (r * y * X)

    return w











