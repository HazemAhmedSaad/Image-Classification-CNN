#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm
import pandas as pd
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import math 
import datetime
import time
import glob as gb


# In[25]:


data_dir = open( 'D:\\selectedProject\\Football_dataset.csv')
x = pd.read_csv(data_dir)


# In[26]:


x.head()


# In[27]:


x.tail()


# In[28]:


print(x.shape)


# In[110]:


import splitfolders as spf


# In[113]:


iput_folder = 'D:\\selectedProject\\football_golden_foot\\football_golden_foot'
spf.ratio(iput_folder, output= "D:\\selectedProject\\football_golden_foot\\new_data",seed=42,ratio=(.8,0,.2),group_prefix=None)


# In[3]:


filesPath = 'D:\\selectedProject\\football_golden_foot\\new_data\\'


# In[4]:


# numper of exampel in each folder (train)

filesPath = 'D:\\selectedProject\\football_golden_foot\\new_data\\'
for folder in os.listdir(filesPath+'train'):
    files = gb.glob(pathname = str(filesPath+'train\\' + folder +'\*.jpg'))
    print(f'For data, found {len(files)} in folder {folder}')


# In[5]:


# Knowning size of all image (train)

size = []
for folder in os.listdir(filesPath+'train\\'):
    files = gb.glob(pathname = str(filesPath+'train\\' + folder +'\*.jpg'))
    for file in files: 
        image =plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()


# In[6]:


# numper of exampel in each folder (test)

for folder in os.listdir(filesPath+'test'):
    files = gb.glob(pathname = str(filesPath+'test\\' + folder +'\*.jpg'))
    print(f'For data, found {len(files)} in folder {folder}')


# In[5]:


# Knowning size of all image (test)

size = []
for folder in os.listdir(filesPath+'test'):
    files = gb.glob(pathname = str(filesPath+'test\\' + folder +'\*.jpg'))
    for file in files: 
        image =plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()


# In[7]:


IMAGE_SIZE = 150


# In[8]:


CATEGORIES = {"cristiano_ronaldo":0, "lionel_messi":1, "mohamed_salah":2, "pele":3, "ronaldinho":4, "zlatan_ibrahimovic":5}
def getCat(n):
    for x, y in CATEGORIES.items():
        if n == y:
            return x


# In[9]:


x_train = []
y_train = []
for folder in os.listdir(filesPath+'train'):
    files = gb.glob(pathname = str(filesPath+'train\\' + folder +'\*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
        x_train.append(list(image_array))
        y_train.append(CATEGORIES[folder])


# In[10]:


x_test = []
y_test = []
for folder in os.listdir(filesPath+'test'):
    files = gb.glob(pathname = str(filesPath+'test\\' + folder +'\*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
        x_test.append(list(image_array))
        y_test.append(CATEGORIES[folder])


# In[11]:


# print(image_array)
print(image_array.shape)


# In[12]:


print(len(x_train) ,"items in x_train")
print(len(x_test) ,"items in x_test")


# In[13]:


plt.figure(figsize=(20,20))
for n,i in enumerate (list(np.random.randint(0,len(x_train),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(x_train[i])
    plt.axis("off")
    plt.title(getCat(y_train[i]))


# In[14]:


plt.figure(figsize=(20,20))
for n,i in enumerate (list(np.random.randint(0,len(x_test),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(x_test[i])
    plt.axis("off")


# In[15]:


plt.figure(figsize=(20,20))
for n,i in enumerate (list(np.random.randint(0,len(x_test),36))):
    plt.subplot(6,6,n+1)
    plt.title(getCat(y_train [i]))


# In[16]:


# print(np.array(x_train).shape)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[17]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[42]:


# create model of CNN before optimization
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])


# In[43]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[44]:


print(model.summary())


# In[45]:


history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split = 0.2)


# In[48]:


test_loss = model.evaluate(x_test, y_test)


# In[20]:


# create model of CNN after optimization

model = tf.keras.Sequential([
    keras.layers.Conv2D(200, (3, 3), activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)),
    keras.layers.Conv2D(150, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(4,4),
    keras.layers.Conv2D(120, (3, 3), activation = 'relu'),
    keras.layers.Conv2D(80, (3, 3), activation = 'relu'),
    keras.layers.Conv2D(50, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(4,4),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(6, activation='softmax'),
])


# In[21]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[22]:


print('Model Details are: ')
print(model.summary())


# In[23]:


history = model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1)


# In[25]:


test_loss = model.evaluate(x_test, y_test)


# In[ ]:


model.save("wh.model")

