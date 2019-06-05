from sgpodv1.model import Sgpod
import os

class Trainer(object):
    def __init__(self, x, labels_count):
        # base_dir = "rawdata/"
        # labels = os.listdir(base_dir)
        # labels_count = len(labels)

        # self.input_size = x.shape[1:]
        self.model = Sgpod(x, labels_count)


    def train(self, x_train, y_train):
        self.model.create_model()
        # for j in range(i):
        self.model.model_train(x_train, y_train)
        # self.model.model_compile(1)
        # self.model.model_fit(x_train, y_train, epochs)
        # self.model.model_compile(2)
        # self.model.model_fit(x_train, y_train, epochs)

    def eval(self, x_test, y_test):
        self.model.sgpod_eval(x_test, y_test)
    
    def predict(self, x):
        print(self.model.sgpod_predict(x))


        

