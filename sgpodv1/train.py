from sgpodv1.model import Sgpod
import os

class Trainer(object):
    def __init__(self, h, w, d):
        base_dir = "./sgpodv1/rawdata/"
        labels = os.listdir(base_dir)
        labels_count = len(labels)

        self.input_size = (h, w)
        self.h, self.w = self.input_size
        try:
            self.dim = d
        except:
            self.dim = 3
            self.model = Sgpod(self.h, self.w, self.dim, 10)
        else:
            self.model = Sgpod(self.h, self.w, self.dim, 10)


    def train(self, x_train, y_train, epochs=10):
        self.model.create_model()
        self.model.model_compile()
        self.model.model_fit(x_train, y_train, epochs)

    def eval(self, x_test, y_test):
        self.model.sgpod_eval(x_test, y_test)

        

