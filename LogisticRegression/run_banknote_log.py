import numpy as np 
import LogisticRegression


def load_data(file_path):
    Data = []

    with open(file_path, 'r') as file:
        for line in file:
            Data.append(line.strip().split(','))
    file.close()

    Data = np.array(Data, dtype='float64')
    X = np.append(Data[:,:-1], np.ones((Data.shape[0], 1)), axis=1)
    y = Data[:, -1]
    y[y == 0] = -1

    return X, y

# load data
X_train, y_train = load_data('./bank-note/train.csv')
X_test, y_test = load_data('./bank-note/test.csv')


# initialize parameters
def r_func(r0, t):
    d = 0.02
    return r0 / (1 + (r0 / d) * t)

T = 100
r0 = 0.01
var_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]


# MLE
print('MLE')
print('var & train_err & test_err')
for v in var_list:
    w = LogisticRegression.logistic_reg_MLE(X_train, y_train, T, r0, r_func, v)
    train_err = LogisticRegression.logistic_reg_test(X_train, y_train, w)
    test_err = LogisticRegression.logistic_reg_test(X_test, y_test, w)
    print(v, '&', train_err, '&', test_err)

print('-----------')


# MAP
print('MAP')
print('var & train_err & test_err')
for v in var_list:
    w = LogisticRegression.logistic_reg_MAP(X_train, y_train, T, r0, r_func, v)
    train_err = LogisticRegression.logistic_reg_test(X_train, y_train, w)
    test_err = LogisticRegression.logistic_reg_test(X_test, y_test, w)
    print(v, '&', train_err, '&', test_err)