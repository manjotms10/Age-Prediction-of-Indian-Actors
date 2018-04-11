import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy.misc import imread, imresize
import matplotlib.pylab as im
import matplotlib.pyplot as plt

def load_dataset():
    x_csv = np.array(pd.read_csv('test.csv'))
    x_train = []
    y_train = []

    for n in range(x_csv.shape[0]):
        x_train.append(imresize(im.imread("Test/" + x_csv[n][0]), (80, 60)) * (1./255))
        
    x_train = np.array(x_train)
    return x_train

x_test = load_dataset()
print(x_test.shape)

model = keras.models.load_model('agePredictionModel.h5')

scores = model.predict(x_test, verbose=2)
plt.imshow(x_test[60])
print("Prediction: " + str(scores[6]))
plt.show()

num_to_class = {
        0 : "YOUNG",
        1 : "MIDDLE",
        2 : "OLD"
        }

predictions = [num_to_class[np.argmax(scores[i], axis=-1)] for i in range(scores.shape[0])]
