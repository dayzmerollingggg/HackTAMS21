import pandas as pd
import PIL
from matplotlib import image
from matplotlib import pyplot
from os import path
import os
from numpy import asarray
from PIL import Image
from os import listdir
from matplotlib import image
from numpy import loadtxt
from numpy import savetxt
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import load
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import joblib
import pickle


photos = load('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/photos.npy')
labels = load('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/labels.npy')
print(photos.shape, labels.shape)

photos = photos/255

X_train, X_test, y_train, y_test = train_test_split(photos, labels, random_state = 100, test_size=0.10)
print((X_train.shape, y_train.shape), (X_test.shape, y_test.shape))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#history = model.fit(X_train, y_train, epochs=5,
                    #validation_data=(X_test, y_test))

model.fit(X_train, y_train, epochs=5)