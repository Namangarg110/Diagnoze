import tensorflow as tf
import keras
from keras.models import load_model

model = load_model('./Saved_model/NN.h5')
model.load_weights('./Saved_model/NN_Weight.h5')

