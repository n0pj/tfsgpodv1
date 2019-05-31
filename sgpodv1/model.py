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



class Sgpod(object):
    def __init__(self, h, w, d, label):
        self.inputs = Input(shape=(h, w, d), name="inputs")
        self.x = UpSampling2D(4)(self.inputs)
        # self.x = self.resblock(self.x, 128, 2)
        # self.x = self.resblock(self.x, 256, 4)
        # self.x = self.resblock(self.x, 512, 8)
        # self.x = self.resblock(self.x, 1024, 2)
        self.x = self.Conv2D_BN_Leaky(32, 2, padding="same")(self.x)
        self.x = self.Inception(self.x, 32)
        self.x = self.Resblock(self.x, 64, 1)
        self.x = self.Inception(self.x, 64)
        self.x = self.Resblock(self.x, 128, 2)
        self.x = self.Inception(self.x, 128)
        self.x = self.Resblock(self.x, 256, 2)
        self.x = GlobalAveragePooling2D()(self.x)
        # self.x = Dense(2979)(self.x)
        # self.x = LeakyReLU()(self.x)
        # self.outputs = Activation("softmax")(self.x)
        self.outputs = Dense(label, activation="softmax")(self.x)

    def Resblock(self, x, n_filter, n_block):
        self.x = ZeroPadding2D((1))(x)
        self.x = self.Conv2D_BN_Leaky(n_filter, (3,3), strides=2)(self.x)
        for i in range(n_block):
            y = compose(
                self.Conv2D_BN_Leaky(n_filter//2, (1,1)),
                self.Conv2D_BN_Leaky(n_filter, (3,3)))(self.x)
            self.x = Add()([self.x, y])
        return self.x

    def Conv2D_BN_Leaky(self, *args, **kwargs):
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        return compose(
            self.L2Conv2D(*args, **no_bias_kwargs),
            BatchNormalization(),
            LeakyReLU(alpha=0.1))

    def L2Conv2D(self, *args, **kwargs):
        l2_conv = {'kernel_regularizer': l2(5e-4)}
        l2_conv['padding'] = 'valid' if kwargs.get('strides')==2 else 'same'
        l2_conv.update(kwargs)
        return Conv2D(*args, **l2_conv)

    def Inception(self, x,  n_filter):
        x1 = self.L2Conv2D(n_filter, 1, strides=2, padding="same")(x)
        x2 = self.L2Conv2D(n_filter, 3, strides=2, padding="same")(x)
        x3 = self.L2Conv2D(n_filter, 5, strides=2, padding="same")(x)
        x4 = MaxPooling2D(2, 2, padding="same")(x)

        y = Concatenate()([x1, x2, x3, x4])
        return y

    def create_model(self):
        self.model = Model(self.inputs, self.outputs, name="network")
        plot_model(self.model, "model.png", show_shapes=True)

    def model_compile(self, i):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer1 = Adam()
        optimizer2 = Adam(lr=1e-3)
        optimizer3 = Adam(lr=1e-4)

        opt_list = [optimizer1, optimizer2, optimizer3]

        self.model.compile(loss=loss_object,
                           optimizer=opt_list[i],
                           metrics=["accuracy"])
        
        # self.model.load_weights("weights/googlenet_weights.h5")

    def model_fit(self, x_train, y_train, i):
        log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        weights_dir = "weights/"
        print(x_train.shape)
        print(y_train.shape)
        logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        epoch_list = [10, 30, 50]
        batch_list = [32, 16, 8]
        iepoc_list = [0, 0, 30]

        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_list[i],
                                 epochs=epoch_list[i],
                                 initial_epoch=iepoc_list[i],
                                 validation_split=0.2,
                                 callbacks=[logging, checkpoint],
                                 )
        self.model.save_weights(weights_dir + 'trained_weights-{}.h5'.format(i))
        self.model.summary()

    def sgpod_eval(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        print("Test loss: {}".format(score[0]))
        print("Test Accu: {}".format(score[1]))
        
        


