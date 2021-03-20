import tensorflow as tf
import os
import keras
from keras.models import load_model
import cv2
import numpy as np
from tqdm import tqdm

model = load_model('tumor_CNN.h5')
model.load_weights('CNN_weights.h5')
predictions = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']
path = './data/Training/glioma_tumor/gg (10).jpg'
img = cv2.imread(path)
img = cv2.resize(img,(70,70))
nor =255
img = img/nor
print('Predictions',predictions[np.argmax(model.predict(img.reshape(1,70,70,3)))])
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()