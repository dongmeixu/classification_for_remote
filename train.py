# set the matplotlib backend so figures can be saved in the backend
# (uncomment the lines below if you are using a headless server)
import matplotlib
matplotlib.use("Agg")
from pyimagesearch.minigooglenet import MiniGoogleNet
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse

