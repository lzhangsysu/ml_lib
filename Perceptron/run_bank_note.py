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

w1 = Perceptron.Perceptron(Data_train, 10, 1.0)
w2 = Perceptron.Perceptron_average(Data_train, 10, 1.0)
w3, c3 = Perceptron.Perceptron_voted(Data_train, 10, 1.0)

print(Perceptron.Perceptron_test(Data_test, w1))
print(Perceptron.Perceptron_test(Data_test, w2))
print(Perceptron.Perceptron_voted_test(Data_test, w3, c3))

