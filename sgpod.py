from sgpodv1.train import Trainer
from sgpodv1.util.utils import convert_images, convert_labels, exchange_data
from sgpodv1.util.utils import make_dataset
import time
import argparse
import tensorflow as tf
h = 32
w = 32
d = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--makedata", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--cifar10", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--predict", type=int)

    args = parser.parse_args()

    if args.makedata:
        exchange_data(h, w)
    elif args.train:
        start = time.time()
        trainer = Trainer(h, w, d)

        x_train, y_train, x_test, y_test = make_dataset()

        if args.epochs:
            trainer.train(x_train, y_train, args.epochs)
            trainer.eval(x_test, y_test)
        else:
            trainer.train(x_train, y_train)
            trainer.eval(x_test, y_test)

        end = time.time() - start
        print("{:.2f}/s".format(end))
    
    elif args.cifar10:
        
        start = time.time()
        trainer = Trainer(h, w, d)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        if args.epochs:
            trainer.train(x_train, y_train, args.epochs)
            trainer.eval(x_test, y_test)
        else:
            trainer.train(x_train, y_train)

        end = time.time() - start
        print("{:.2f}/s".format(end))
    
    else:
        try:
            raise Exception
        except:
            print("please set argument.")


if __name__ == "__main__":
    main()

