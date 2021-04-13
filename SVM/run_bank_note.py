import numpy as np
import SVM

### read file into numpy array ###
def read_data(file_path, bias=True):
    Data = []

    with open(file_path, 'r') as file:
        for line in file:
            Data.append(line.strip().split(','))

    Data = np.array(Data, dtype='float64')

    X = Data[:, :-1]
    y = Data[:, -1].astype(int)
    y[y == 0] = -1  # convert y label

    return X, y


X_train, y_train = read_data('c:/Users/Li/Desktop/Utah/ML/ML_lib/SVM/bank-note/train.csv')
X_test, y_test = read_data('c:/Users/Li/Desktop/Utah/ML/ML_lib/SVM/bank-note/test.csv')


C_vals =  [100.0/873, 500.0/873, 700.0/873]

"""
### Q2: SVM with stochastic gradient descent
def schedule_A(gamma_0, t, d=0.1):
    return gamma_0 / (1 + (gamma_0 / d) * t)

def schedule_B(gamma_0, t):
    return gamma_0 / (1 + t)

# Q2A schedule A
print('SVM primal sgd with schedule A')
for C in C_vals:
    w = SVM.SVM_primal_sgd(X_train, y_train, epochs=100, C=C, gamma=0.01, schedule_func=schedule_A)
    print('C:', C, 'weights:', w)
    err_train = SVM.SVM_primal_test(X_train, y_train, w)
    err_test = SVM.SVM_primal_test(X_test, y_test, w)
    print('training error:', err_train, 'test error', err_test)


# Q2A schedule B
print('\nSVM primal sgd with schedule B')
for C in C_vals:
    w = SVM.SVM_primal_sgd(X_train, y_train, epochs=100, C=C, gamma=0.01, schedule_func=schedule_B)
    print('C:', C, 'weights:', w)
    err_train = SVM.SVM_primal_test(X_train, y_train, w)
    err_test = SVM.SVM_primal_test(X_test, y_test, w)
    print('training error:', err_train, 'test error', err_test)
"""


### Q3 SVM dual and kernal

# Q3A SVM dual
print('\nSVM dual')
for C in C_vals:
    w, b = SVM.SVM_dual(X_train, y_train, C)
    print('C:', C, 'weights:', w, 'bias:', b)
    err_train = SVM.SVM_dual_test(X_train, y_train, w, b)
    err_test = SVM.SVM_dual_test(X_test, y_test, w, b)
    print('training error:', err_train, 'test error', err_test)