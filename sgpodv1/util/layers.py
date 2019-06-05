import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, AveragePooling2D, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from sgpodv1.util.utils import compose
from functools import wraps

def Resblock(x, n_filter, n_block):
    x = ZeroPadding2D(((1,0), (1,0)))(x)
    x = Conv2DBNLeaky(n_filter, 3, strides=2)(x)
    for i in range(n_block):
        y = compose(
            Conv2DNB(n_filter//2, 1),
            Conv2DBNLeaky(n_filter//2, 3),
            Conv2DBNLeaky(n_filter, 1))(x)
        x = Add()([x, y])
    return x

def XBlockv1(x, n_filter, k):
    x1 = compose(
        Conv2DBNLeaky(n_filter//2, 1),
        Conv2DBNLeaky(n_filter, 3),
        AveragePooling2D(2, 2))(x[1])

    x2 = compose(
        Conv2DBNLeaky(n_filter//2, 1),
        Conv2DBNLeaky(n_filter, 3),
        AveragePooling2D(2, 2))(x[1])

    x1 = Add()([x1, x2])

    # x2 = compose(
    #     Conv2DBNLeaky(n_filter//2, 1),
    #     Conv2DBNLeaky(n_filter, k, strides=2, padding="same"))(x[0])
    return (x1, x2)

def XBlockv2(x, n_filter, k):
    x1 = Resblock(x[1], n_filter, k)
    x2 = Resblock(x[0], n_filter, k)
    return (x1, x2)

def Add2(x):
    # x1 = compose(
    #     Conv2DBNLeaky(n_filter, 2))(x[1])
    # x2 = compose(
    #     Conv2DBNLeaky(n_filter, 2))(x[0])

    return Add()([x[0], x[1]])

def Concate(x):
    return Concatenate()([x[0], x[1]])

def Distensor(x):
    x1 = x
    x2 = x
    return (x1, x2)

def Conv2DBNLeaky(*args, **kwargs):
    return compose(
        Conv2DL2(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def Conv2DNB(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        Conv2D(*args, **kwargs))

def Conv2DL2(*args, **kwargs):
    l2_conv = {'kernel_regularizer': l2(5e-4)}
    l2_conv['padding'] = 'valid' if kwargs.get('strides')==2 else 'same'
    l2_conv.update(kwargs)
    return Conv2D(*args, **l2_conv)

def Inception(x,  n_filter):
    x1 = compose(
        Conv2DBNLeaky(n_filter, 1))(x)
    x2 = compose(
        Conv2DBNLeaky(n_filter, 3))(x)
    x3 = compose(
        Conv2DBNLeaky(n_filter, 5))(x)
    x4 = compose(
        MaxPooling2D(2, 2, padding="same"))(x)

    x = Concatenate()([x1, x2, x3, x4])
    return x
