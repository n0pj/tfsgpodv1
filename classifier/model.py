import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, AveragePooling2D, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sgpodv1.util.utils import compose, lr_schedule
from sgpodv1.util.layers import Resblock, Conv2DNB, Conv2DBNLeaky, Inception
from sgpodv1.util.layers import XBlockv1, Add2, Concate, Distensor, XBlockv2
from functools import wraps



class Sgpod(object):
    def __init__(self, x, label):
        self.inputs = Input(shape=x.shape, name="inputs")
        self.x = Conv2DBNLeaky(32, 2, padding="same")(self.inputs)
        # self.x = darknet_body()(self.inputs)
        self.x = Distensor(self.x)
        self.x = XBlockv2(self.x, 32, 1)
        self.x = Add2(self.x)
        self.x = Distensor(self.x)
        self.x = XBlockv2(self.x, 64, 1)
        self.x = Add2(self.x)
        self.x = Distensor(self.x)
        self.x = XBlockv2(self.x, 128, 2)
        self.x = Add2(self.x)
        self.x = GlobalAveragePooling2D()(self.x)
        self.outputs = Dense(label, activation="softmax")(self.x)

    def create_model(self):
        self.model = Model(self.inputs, self.outputs, name="network")
        plot_model(self.model, "model.png", show_shapes=True)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        op1 = Adam(lr=lr_schedule(0))
        # op2 = RMSprop(lr=lr_schedule(0))

        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=op1,
                           metrics=["accuracy"])
        return self.model

    def model_train(self, x_train, y_train):
        log_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        weights_dir = "weights/"
        print(x_train.shape)
        print(y_train.shape)
        logging = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(weights_dir + 'val_loss{val_loss:.3f}-loss{loss:.3f}-ep{epoch:03d}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        epoch_list = [5, 10, 15, 20, 25]
        iepoc_list = [0, 5, 10, 15, 20]
        batch_list = [32, 32, 16, 16, 16]

        epoch_list2 = [5, 15, 25, 30, 35]
        iepoc_list2 = [0, 5, 15, 25, 30]
        history = self.model.fit(x_train, y_train,
                                 epochs=40,
                                 initial_epoch=0,
                                 batch_size=16,
                                 validation_split=0.2,
                                 callbacks=[logging, checkpoint, lr_scheduler, lr_reducer],
                                 )

        # if False:
        #     for i in range(len(self.model.layers)):
        #         self.model.layers[i].trainable = True 

        #     self.model.compile(loss=loss_object,
        #                        optimizer=op1,
        #                        metrics=["accuracy"])

        #     history = self.model.fit(x_train, y_train,
        #                              epochs=25,
        #                              initial_epoch=20,
        #                              batch_size=16,
        #                              validation_split=0.2,
        #                              callbacks=[logging, checkpoint, lr_scheduler, lr_reducer],
        #                              )

        self.model.save_weights(weights_dir + 'trained_weights.h5')
        self.model.save("model.h5")
        self.model.summary()

    def sgpod_eval(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        print("Test loss: {}".format(score[0]))
        print("Test Accu: {}".format(score[1]))
        
    def sgpod_predict(self, x):
        # model = self.create_model()
        model = load_model("model.h5")

        # weights_path = "weights/val_loss0.806-loss0.476-ep024.h5"
        # model.load_weights(weights_path)
        from PIL import Image
        # x = Image.open(x)
        x = np.asarray(x)
        # x = x/255
        x = np.expand_dims(x, 0)
        img = []
        img.append(x)

        result = model.predict(img)[0]
        pre = result.argmax()
        per = int(result[pre]*100)
        print(per, pre, result)



