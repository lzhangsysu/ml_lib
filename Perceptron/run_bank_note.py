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


### standard perceptron ###
# train to learn weights
w_std = Perceptron.Perceptron_standard(Data_train, T=10, r=0.1)
print("standard Perceptron: weight vector", w_std)

# run on test dataset
err_std = Perceptron.Perceptron_test(Data_test, w_std)
print("standard Perceptron: test error", err_std)


### voted perceptron ###
# train to learn weights and votes
W_vt, C_vt = Perceptron.Perceptron_voted(Data_train, T=10, r=0.1)

outFile = open('percep_vt_out.txt', 'w')
outFile.write("weight_vector\tvotes\n")
for w_c in zip(W_vt, C_vt):
    w = w_c[0]
    v = w_c[1]
    outFile.write(str(w) + '\t' + str(v) + '\n')

outFile.close()

# run on test dataset
err_vt = Perceptron.Perceptron_voted_test(Data_test, W_vt, C_vt)
print("voted Perceptron: test error", err_vt)


### average perceptron ###
# train to learn weights
w_avg = Perceptron.Perceptron_average(Data_train, T=10, r=0.1)
print("averaged Perceptron: weight vector", w_avg)

# run on test dataset
err_avg = Perceptron.Perceptron_test(Data_test, w_avg)
print("averaged Perceptron: test error", err_avg)
