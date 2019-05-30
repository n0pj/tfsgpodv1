import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Activation, MaxPooling2D
from tensorflow.keras import Model

class Sgpod(Model):
    
    def __init__(self):
        super(Sgpod, self).__init__()
        self.flatten = Flatten()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.conv2 = Conv2D(32, 3, activation="relu")
        # self.mpool1 = MaxPooling2D(2)
        self.dense2 = Dense(128, activation="relu")
        self.dense3 = Dense(10, activation="softmax")

    def call(self, x):
        print(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        # x = self.mpool1(x)
        x = self.dense2(x)
        return self.dense3(x)

    