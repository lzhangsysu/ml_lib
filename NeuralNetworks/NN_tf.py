import tensorflow as tf 
from tensorflow import keras
import numpy as np

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

width = 10    # hiden layer width
depth = 3

layers = [keras.layers.Flatten(input_shape=(1,5))]
layers.extend([keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros') for _ in range(depth-2)])
layers.append(keras.layers.Dense(2, activation=tf.nn.softmax))

model = keras.Sequential(layers)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(Data_train, Labels_train, epochs= 20)

test_loss, test_acc = model.evaluate(Data_test, Labels_test)

print(test_loss, test_acc)
