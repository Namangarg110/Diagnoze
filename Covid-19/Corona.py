#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.listdir()


# In[3]:


os.chdir('data')


# In[4]:


os.listdir()


# In[5]:


os.chdir('..')


# In[6]:


os.listdir()


# In[20]:


train_path = 'data/train'
test_path = 'data/test'
valid_path = 'data/valid'


# In[9]:


print(os.listdir(train_path))
print(os.listdir(test_path))


# In[18]:


len(os.listdir(test_path+'/covid'))


# In[21]:


len(os.listdir(valid_path+'/covid'))


# In[25]:


len(os.listdir(train_path+'/non'))


# In[10]:


mobile = tf.keras.applications.mobilenet.MobileNet()


# In[31]:


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_path, target_size=(224,224), batch_size=10)


# In[32]:


valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path, target_size=(224,224), batch_size=10)


# In[33]:


test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    test_path, target_size=(224,224), batch_size=10, shuffle=False)


# In[34]:


mobile.summary()


# In[35]:


mobile.count_params()


# In[36]:


x = mobile.layers[-6].output
output = Dense(units=2, activation='sigmoid')(x)


# In[37]:


model = Model(inputs=mobile.input, outputs=output)


# In[38]:


model.summary()


# In[39]:


for layer in model.layers[:-23]:
    layer.trainable = False


# In[40]:


model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# In[41]:


model.fit(x=train_batches, steps_per_epoch=len(train_batches), validation_data=valid_batches, 
          validation_steps=len(valid_batches), epochs=15, verbose=2)


# In[42]:


test_labels = test_batches.classes


# In[44]:


test_labels


# In[45]:


predictions = model.predict(x=test_batches, verbose=2)


# In[49]:


pred = predictions.argmax(axis=1)


# In[50]:


pred


# In[51]:


cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))


# In[52]:


cm


# In[53]:


from tensorflow.keras.models import model_from_json


# In[54]:


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# In[ ]:




