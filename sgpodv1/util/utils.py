import numpy as np
import tensorflow as tf
from PIL import Image
import os, glob
from tqdm import tqdm, trange
from functools import reduce

def convert_images(images):
    cdata = images / 255
    return cdata

def convert_labels(labels, labels_count):
    labels = tf.one_hot(labels, labels_count)
    return labels

def exchange_data(h, w):
    base_dir = "./sgpodv1/rawdata/"
    labels = os.listdir(base_dir)
    labels_count = len(labels)
    
    split_p = 0.2

    x_train, x_test, y_train, y_test = [], [], [], []

    for i, label in enumerate(labels):
        image_dir = base_dir + label
        files = glob.glob(image_dir + "/*.jpg")
        file_count = len(files)
        template = "\nLabel {}, File count {}\n"
        print(template.format(label, file_count))

        for idx, image in tqdm(enumerate(files)):
            img = Image.open(image)
            img = img.convert("RGB")
            img = img.resize((h, w))
            np_img = np.asarray(img)

            testset = file_count * split_p

            if idx <= testset:
                x_train.append(np_img)
                y_train.append(i)
                
                for angle in range(-40, 40, 20):
                    rotate_img = img.rotate(angle)
                    np_img = np.asarray(rotate_img)
                    x_train.append(np_img)
                    y_train.append(i)
            else:
                
                for angle2 in range(-40, 40, 20):
                    rotate_img = img.rotate(angle2)
                    np_img = np.asarray(rotate_img)
                    x_test.append(np_img)
                    y_test.append(i)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    dataset = (x_train, y_train, x_test, y_test)
    np.save("sgpodv1/dataset.npy", dataset)
    print("\nDatasets created done.")

def make_dataset():
    base_dir = "./sgpodv1/rawdata/"
    labels = os.listdir(base_dir)
    labels_count = len(labels)

    x_train, y_train, x_test, y_test = np.load("./sgpodv1/dataset.npy", allow_pickle=True)
    x_train = convert_images(x_train)
    y_train = convert_labels(y_train, labels_count)
    x_test = convert_images(x_test)
    y_test = convert_labels(y_test, labels_count)
    return x_train, y_train, x_test, y_test

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
