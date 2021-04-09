import numpy as np
import SVM

### read file into numpy array ###
Data_train = []
Data_test = []

with open('./bank-note/train.csv', 'r') as train_file:
    for line in train_file:
        Data_train.append(line.strip().split(','))
train_file.close()

with open('./bank-note/test.csv', 'r') as test_file:
    for line in test_file:
        Data_test.append(line.strip().split(','))
test_file.close()

Data_train = np.array(Data_train, dtype='float64')
Data_test = np.array(Data_test, dtype='float64')


C_vals =  [100.0/873, 500.0/873, 700.0/873]


def r_rate_A(r0, t, d=0.002):
    return r0 / (1 + (r0 / d) * t)

def r_rate_B(r0, t):
    return r0 / (1 + t)

for C in C_vals:
    w = SVM.SVM_sgd(Data_train, C, 100, 0.001, r_rate_A)
    w /= np.linalg.norm(w)
    err_train = SVM.SVM_test(Data_train, w)
    err_test = SVM.SVM_test(Data_test, w)
    print(err_train, err_test)