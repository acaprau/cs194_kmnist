# utility functions for benchmarks
import numpy as np
import os

from PIL import Image


#from tensorflow.keras.callbacks import Callback
#import wandb

# extend Keras callback to log benchmark-specific key, "kmnist_val_acc"
#class KmnistCallback(Callback):
#  def on_epoch_end(self, epoch, logs={}):
#    if "val_accuracy" in logs:
#        # latest version of tensorflow
#        #wandb.log({"kmnist_val_acc" : logs["val_accuracy"]}, commit=False)
#        pass
#    elif "val_acc" in logs:
#        # older version of tensorflow
#        #wandb.log({"kmnist_val_acc" : logs["val_acc"]}, commit=False)
#        pass
#    else:
#        raise Exception("Keras logs object missing validation accuracy")

# load data file into array
def load(f):
    return np.load(f)['arr_0']

def load_train_data(datadir):
  x_train = load(os.path.join(datadir, 'kmnist-train-imgs.npz'))
  y_train = load(os.path.join(datadir, 'kmnist-train-labels.npz'))
  return x_train, y_train

def load_test_data(datadir):
  x_test = load(os.path.join(datadir, 'kmnist-test-imgs.npz'))
  y_test = load(os.path.join(datadir, 'kmnist-test-labels.npz'))
  return x_test, y_test

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

def write_numpy_image(img, filepath='img.png'):
    img = Image.fromarray(img, mode='L')
    img.save(filepath)

def write_pillow_image(img, filepath='img.png'):
    img.save(filepath)

def read_image_to_pillow(filepath):
    img = Image.open(filepath)
    return img

def read_pillow_from_dataset(dataset='test', element=0):
    if dataset == 'test':
        data = load('dataset/kmnist-test-imgs.npz')
    else:
        data = load('dataset/kmnist-train-imgs.npz')
    img = data[element]
    img = Image.fromarray(img, mode='L')
    return img
