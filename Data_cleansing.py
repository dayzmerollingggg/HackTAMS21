#!/usr/bin/env python
# coding: utf-8

# In[2]:


#install all the packages 
import pandas as pd
import PIL
from matplotlib import image
from matplotlib import pyplot
from os import path
import os
from numpy import asarray
from PIL import Image
# load all images in a directory
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


# In[2]:


photos = []
labels = []

for filename in listdir('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/2. Mild (or early) NPDR'):
    img_data = os.path.join('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/2. Mild (or early) NPDR', filename)
    image = load_img(img_data, target_size=(512, 512))
    photo = img_to_array(image)
    output = 1
    photos.append(photo)      
    labels.append(output)
for filename in listdir('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/3. Moderate NPDR'):
    img_data = os.path.join('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/3. Moderate NPDR', filename)
    image = load_img(img_data, target_size=(512, 512))
    photo = img_to_array(image)
    output = 2
    photos.append(photo)      
    labels.append(output)
for filename in listdir('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/4. Severe NPDR'):
    img_data = os.path.join('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/4. Severe NPDR', filename)
    image = load_img(img_data, target_size=(512, 512))
    photo = img_to_array(image)
    output = 3
    photos.append(photo)      
    labels.append(output)
for filename in listdir('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/5. Very Severe NPDR'):
    img_data = os.path.join('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/5. Very Severe NPDR', filename)
    image = load_img(img_data, target_size=(512, 512))
    photo = img_to_array(image)
    output = 4
    photos.append(photo)      
    labels.append(output)
for filename in listdir('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/6. PDR'):
    img_data = os.path.join('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/6. PDR', filename)
    image = load_img(img_data, target_size=(512, 512))
    photo = img_to_array(image)
    output = 5
    photos.append(photo)      
    labels.append(output)
for filename in listdir('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/7. Advanced PDR'):
    img_data = os.path.join('/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/7. Advanced PDR', filename)
    image = load_img(img_data, target_size=(512, 512))
    photo = img_to_array(image)
    output = 6
    photos.append(photo)      
    labels.append(output)


# In[3]:


photos = photos/255


# In[8]:


save("/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/photos.npy", photos)
save("/Users/Shreya.A/PycharmProjects/HackTAMS/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02/labels.npy", labels)

