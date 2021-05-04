import tensorflow as tf 
from tensorflow import keras
import numpy as np


# read data from file
def load_data(file_path):
    Data = []
    Labels = []

    with open(file_path, 'r') as file:
        for line in file:
            cols = line.strip().split(',')
            Data.append([float(c) for c in cols[:-1]])
            if cols[-1] == '1':
                Labels.append(1)
            else:
                Labels.append(0)

    file.close()

    Data = np.array(Data)
    Data = np.append(Data, np.ones((Data.shape[0], 1)), axis=1)
    Input = []
    for row in Data:
        wrap = [row]
        Input.append(wrap)
    Input = np.array(Input)
    Labels = np.array(Labels)
    return Input, Labels

Data_train, Labels_train = load_data('./bank-note/train.csv')
Data_test, Labels_test = load_data('./bank-note/test.csv')


# parameters
depth_arr = [3, 5, 9]
width_arr = [5, 10, 25, 50, 100]


# tanh activation
tanh_file = open('tanh_output.txt', 'w')
tanh_file.write('depth, width, train_err, test_err\n')
tanh_file.close()
tanh_file = open('tanh_output.txt', 'a')

for depth in depth_arr:
    for width in width_arr:
        # build layers
        layers = [keras.layers.Flatten(input_shape=(1,5))]
        layers.extend([keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros') for _ in range(depth-2)])
        layers.append(keras.layers.Dense(2, activation=tf.nn.softmax))
        
        # build model
        model = keras.Sequential(layers)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # train model
        model.fit(Data_train, Labels_train, epochs=20)
        
        # evaluate model
        train_loss, train_acc = model.evaluate(Data_train, Labels_train)
        test_loss, test_acc = model.evaluate(Data_test, Labels_test)
        
        tanh_file.write(str(depth) + ',' + str(width) + ',' + str(round(1-train_acc, 4)) + ',' + str(round(1-test_acc, 4)) + '\n')

tanh_file.close()


# relu activation
relu_file = open('relu_output.txt', 'w')
relu_file.write('depth, width, train_err, test_err\n')
relu_file.close()
relu_file = open('relu_output.txt', 'a')

for depth in depth_arr:
    for width in width_arr:
        # build layers
        layers = [keras.layers.Flatten(input_shape=(1,5))] 
        layers.extend([keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros') for _ in range(depth-1)]) # hidden
        layers.append(keras.layers.Dense(2, activation=tf.nn.softmax)) # output
        
        # build model
        model = keras.Sequential(layers)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # train model
        model.fit(Data_train, Labels_train, epochs=20)
        
        # evaluate model
        train_loss, train_acc = model.evaluate(Data_train, Labels_train)
        test_loss, test_acc = model.evaluate(Data_test, Labels_test)
        
        relu_file.write(str(depth) + ',' + str(width) + ',' + str(round(1-train_acc, 4)) + ',' + str(round(1-test_acc, 4)) + '\n')

relu_file.close()
