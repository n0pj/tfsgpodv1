import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import datetime
import matplotlib.pyplot as plt
from sgpodv1.util.utils import compose
from functools import wraps



class Sgpod(object):
    def __init__(self, h, w, d, label):
        self.inputs = Input(shape=(h, w, d), name="inputs")
        self.x = UpSampling2D(3)(self.inputs)
        self.x = self.resblock(self.x, 128, 2)
        self.x = AveragePooling2D(2, 2, padding="same")(self.x)
        self.x = self.resblock(self.x, 256, 4)
        self.x = AveragePooling2D(2, 2, padding="same")(self.x)
        self.x = self.resblock(self.x, 512, 8)
        self.x = AveragePooling2D(2, 2, padding="same")(self.x)
        self.x = self.resblock(self.x, 1024, 2)
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = Dense(1024)(self.x)
        self.x = LeakyReLU()(self.x)
        self.outputs = Dense(label, activation="softmax")(self.x)

    def resblock(self, x, n_filter, n_block):
        self.x = ZeroPadding2D((1))(x)
        self.x = self.Conv2D_BN_Leaky(n_filter, (3,3), strides=(2,2))(self.x)
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
        l2_conv['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
        l2_conv.update(kwargs)
        return Conv2D(*args, **l2_conv)


    def create_model(self):
        self.model = Model(self.inputs, self.outputs, name="network")
        plot_model(self.model, "model.png", show_shapes=True)

    def model_compile(self):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        optimizer2 = tf.optimizers.RMSprop(lr=1e-4, decay=1e-6)

        self.model.compile(loss=loss_object,
                           optimizer=optimizer,
                           metrics=["accuracy"])
        
        # self.model.load_weights("weights/ep006-loss0.359-val_loss1.133.h5")


    def model_fit(self, x_train, y_train, epoch=10):
        log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        weights_dir = "weights/"
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        print(x_train.shape)
        print(y_train.shape)
        logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        history = self.model.fit(x_train, y_train,
                                 batch_size=32,
                                 epochs=epoch,
                                 validation_split=0.2,
                                 callbacks=[logging, checkpoint],
                                 )
        self.model.save_weights(log_dir + 'trained_weights_stage_final.h5')
        self.model.summary()

    def sgpod_eval(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        print("Test loss: {}".format(score[0]))
        print("Test Accu: {}".format(score[1]))
        
        


