#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras import regularizers
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras import regularizers
from keras.models import load_model
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[15]:


os.listdir('./data/Training')


# In[16]:


loc0 = './data/Training/no_tumor'
loc1 = './data/Training/pituitary_tumor'
loc2 = './data/Training/meningioma_tumor'
loc3 = './data/Training/glioma_tumor'


# In[17]:


import cv2
from tqdm import tqdm
features = []

for img in tqdm(os.listdir(loc0)):
    f = cv2.imread(os.path.join(loc0,img))
    fr = cv2.resize(f,(256,256))
    features.append(fr)
    
for img in tqdm(os.listdir(loc1)):
    f = cv2.imread(os.path.join(loc1,img))
    fr = cv2.resize(f,(256,256))
    features.append(fr)
    
for img in tqdm(os.listdir(loc2)):
    f = cv2.imread(os.path.join(loc2,img))
    fr = cv2.resize(f,(256,256))
    features.append(fr)
    
for img in tqdm(os.listdir(loc3)):
    f = cv2.imread(os.path.join(loc3,img))
    fr = cv2.resize(f,(256,256))
    features.append(fr)


# In[ ]:





# In[18]:


labels = []
for img in tqdm(os.listdir(loc0)):
    labels.append(0) #no tumor 
    
for img in tqdm(os.listdir(loc1)):
    labels.append(1) #pituitary_tumor
    
for img in tqdm(os.listdir(loc2)):
    labels.append(2) #meningioma_tumor
    
for img in tqdm(os.listdir(loc3)):
    labels.append(3) #glioma_tumor


# In[19]:


import numpy as np
X = np.array(features)
Y = np.array(labels)


# In[20]:


print(X.shape)
print(Y.shape)


# In[21]:


#Define a CNN model to make predictions
weight_decay = 1e-4
model = Sequential()

#1st Convolutional Layer
model.add(Conv2D(32,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay) ,input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

#2nd Convolutional Layer
model.add(Conv2D(64,(3,3), kernel_regularizer=regularizers.l2(weight_decay) , padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#3rd Convolutional Layer
model.add(Conv2D(128,(3,3), kernel_regularizer=regularizers.l2(weight_decay) , padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

#4th Convolutional Layer
model.add(Conv2D(256,(3,3), kernel_regularizer=regularizers.l2(weight_decay) , padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#5th Convolutional Layer
model.add(Conv2D(512,(3,3), kernel_regularizer=regularizers.l2(weight_decay) , padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

#6th Convolutional Layer
model.add(Conv2D(1024,(3,3), kernel_regularizer=regularizers.l2(weight_decay) , padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#Fully connected layer
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4,activation='softmax'))


# In[22]:


model.summary()


# In[23]:


#Compile your model

from keras import optimizers
from keras import metrics

sgd = optimizers.SGD(0.1)

model.compile(optimizer=sgd,
             loss = 'binary_crossentropy',
             metrics=['accuracy'])


# In[24]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.8)


# In[25]:


xtrain_n = xtrain/xtrain.max()
xtest_n = xtest/xtest.max()


# In[26]:


ytrain_h = np_utils.to_categorical(ytrain)
ytest_h = np_utils.to_categorical(ytest)


# In[ ]:


model.fit(xtrain_n,ytrain_h,
         epochs=100,
         validation_data=(xtest_n,ytest_h))


# In[27]:


model=load_model('tumor_CNN.h5')


# In[28]:


model.evaluate(xtrain_n,ytrain_h)


# In[29]:


model.evaluate(xtest_n,ytest_h)


# In[30]:


ytest_pred = []
preds = model.predict(xtest_n)
for p in preds:
    ytest_pred.append(np.argmax(p))


# In[31]:


from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,ytest_pred)


# In[32]:


xtrain_n.shape


# In[47]:


xtest_n[90].shape


# In[57]:


predictions = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']
x=121

print('Predictions',predictions[np.argmax(model.predict(xtest_n[i].reshape(1,256,256,3)))])
print('Actual',predictions[ytest[i]])


import matplotlib.pyplot as plt
plt.imshow(xtest_n[90])
plt.show()


# In[58]:


path = './data/Training/glioma_tumor/gg (10).jpg'
img = cv2.imread(path)
img = cv2.resize(img,(256,256))
img = img/xtrain.max()
print('Predictions',np.argmax(model.predict(img.reshape(1,256,256,3))))

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()


# In[59]:


xtrain.max()


# In[ ]:




