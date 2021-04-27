import numpy as np 
import LogisticRegression

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
Data_train[:,-1][Data_train[:,-1] == 0] = -1
Data_test[:,-1][Data_test[:,-1] == 0] = -1

def r_func(r0, t):
    d = 0.1
    return r0 / (1 + (r0 / d) * t)

T = 100
var_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]


for v in var_list:
    w = LogisticRegression.logistic_reg_MLE(Data_train, T, 0.05, r_func, v)
    print('train', LogisticRegression.logistic_reg_test(Data_train, w))
    print('test', LogisticRegression.logistic_reg_test(Data_test, w))