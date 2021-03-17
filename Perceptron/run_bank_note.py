import numpy as np
import Perceptron


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

print(Perceptron.Perceptron(Data_train, 10, 1.0))