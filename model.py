import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy.misc import imread, imresize
import matplotlib.pylab as im
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Concatenate
from keras.layers import Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping

def get_model():
    input_layer = Input(shape=(80, 60, 3))
    x = Conv2D(32, kernel_size = (3,3), strides=(1,1), padding = 'SAME', activation = 'relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    conv2 = Conv2D(16, kernel_size = (3,3), activation = 'relu', padding = 'SAME')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, kernel_size = (5,5), activation = 'relu', padding = 'SAME')(pool2)
    drop = Dropout(0.2)(conv3)
    conv4 = Conv2D(16, kernel_size = (3,3), activation = 'relu', padding = 'SAME')(drop)
    conv5 = Conv2D(32, kernel_size = (1,1), activation = 'relu', padding = 'SAME')(conv4)
    flat = Flatten()(conv5)
    hidden1 = Dense(80, activation='relu')(flat)
    output = Dense(3, activation='softmax')(hidden1)

    model = Model(inputs = input_layer, outputs = output)
    return model

def load_dataset():
    x_csv = np.array(pd.read_csv('train.csv'))
    x_train = []
    y_train = []
    class_to_num = {
        "YOUNG" : 0,
        "MIDDLE" : 1,
        "OLD" : 2
        }

    for n in range(x_csv.shape[0]):
        x_train.append(imresize(im.imread("Train/" + x_csv[n][0]), (80, 60)) * (1./255))
        y_train.append(class_to_num[str(x_csv[n][1])])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.eye(3)[y_train]
    return x_train, y_train

model = get_model()
x_train, y_train = load_dataset()
print(x_train.shape)
print(y_train.shape)

model.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = ["accuracy"])

hist = model.fit(x_train, y_train, epochs = 20, verbose=2, batch_size = 80, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
print()
print(hist.history)
print(model.summary)
model.save('agePredictionModel.h5', overwrite=True)
