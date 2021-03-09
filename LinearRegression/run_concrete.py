import numpy as np
import LMS


# load data from file
Data_train = []
Data_test = []

with open('./concrete/train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        Data_train.append(terms)

train_file.close()

with open('./concrete/test.csv', 'r') as test_file:
    for line in test_file:
        terms = line.strip().split(',')
        Data_test.append(terms)

test_file.close()


# obtain X_train, y_train, X_test, y_test
Data_train = np.array(Data_train, dtype='float64')
Data_test = np.array(Data_test, dtype='float64')

X_train = np.append(np.ones((Data_train.shape[0], 1), dtype='float64'), Data_train[:, 0:7], 1)
y_train = Data_train[:,7].flatten()

X_test = np.append(np.ones((Data_test.shape[0], 1), dtype='float64'), Data_test[:, 0:7], 1)
y_test = Data_test[:,7].flatten()


### Batch gradient descent ###
# GD on training data
lr = 0.0125
w_train_batch, costs_train_batch = LMS.batch_gradient_descent(X_train, y_train, lr)
print('Learned weight vector:', w_train_batch)
LMS.plot_cost(costs_train_batch, "Costs in Batch Gradient Descent (learning rate {})".format(lr))
# calculate cost on test data using learned weights
w_test_batch = LMS.calc_cost(w_train_batch, X_test, y_test)
print('cost on test with GD', w_test_batch)


### Stochastic gradient descent ###
# GD on training data
lr = 0.002
w_train_sto, costs_train_sto = LMS.stochastic_gradient_descent(X_train, y_train, lr)
print('Learned weight vector:', w_train_sto)
LMS.plot_cost(costs_train_sto, "Costs in Stochastic Gradient Descent (learning rate {})".format(lr))
# calculate cost on test data using learned weights
w_test_sto = LMS.calc_cost(w_train_sto, X_test, y_test)
print('cost on test with SGD', w_test_sto)


### Optimal weights ###
X = np.array(X_train).transpose()
y = y_train
a = np.linalg.inv(np.matmul(X, X.transpose()))
b = np.matmul(a, X)
w_opt = np.matmul(b, y)
print(w_opt)