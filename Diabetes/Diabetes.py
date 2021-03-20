#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter


# In[2]:


data = pd.read_csv("diabetes75pc_100_times.csv")


# In[3]:


data["Outcome"].value_counts()


# In[4]:


data.info()


# In[5]:


X = data.drop("Outcome", axis=1)
y = data["Outcome"]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[7]:


X_train.shape


# In[8]:


columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


# In[9]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)


# In[10]:


X_train = pd.DataFrame(scaled_data, columns=columns)


# In[11]:


X_train


# In[12]:


scaled_test = scaler.transform(X_test)
X_test = pd.DataFrame(scaled_test, columns=columns)


# In[13]:


X_test


# In[14]:


tree = DecisionTreeClassifier(random_state=10)


# In[15]:


tree.fit(X_train, y_train)


# In[16]:


pred = tree.predict(X_test)


# In[17]:


tree.score(X_train, y_train)


# In[18]:


accuracy_score(y_test, pred)


# In[19]:


confusion_matrix(y_test, pred)


# In[20]:


forest = RandomForestClassifier(random_state=10)


# In[21]:


forest.fit(X_train, y_train)


# In[22]:


pred_f = forest.predict(X_test)


# In[23]:


accuracy_score(y_test, pred_f)


# In[24]:


confusion_matrix(y_test, pred_f)


# In[ ]:


svm_clf = SVC(random_state=10)
svm_clf.fit(X_train, y_train)


# In[ ]:


pred_s = svm_clf.predict(X_test)


# In[ ]:


accuracy_score(y_test, pred_s)


# In[ ]:


confusion_matrix(y_test, pred_s)


# In[ ]:


# the numbers before smote
num_before = dict(Counter(y))

#perform smoting

# define pipeline
over = SMOTE(sampling_strategy=0.8)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_smote, y_smote = pipeline.fit_resample(X, y)


#the numbers after smote
num_after =dict(Counter(y_smote))


# In[ ]:


num_before


# In[ ]:


num_after


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[ ]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)


# In[ ]:


X_train = pd.DataFrame(scaled_data, columns=columns)


# In[ ]:


scaled_test = scaler.transform(X_test)
X_test = pd.DataFrame(scaled_test, columns=columns)


# In[ ]:


forest_s = RandomForestClassifier(random_state=5)


# In[ ]:


forest_s.fit(X_train, y_train)
pred_s = forest_s.predict(X_test)


# In[ ]:


accuracy_score(y_test, pred_s)


# In[ ]:


confusion_matrix(y_test, pred_s)


# ### Neural Network

# In[25]:


import pandas
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[32]:


model = Sequential([
    Dense(8, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[33]:


model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[34]:


model.fit(X_train, y_train, epochs=100)


# In[35]:


p = model.predict(X_test)


# In[36]:


pred = np.round(p)


# In[37]:


accuracy_score(y_test, pred)


# In[39]:


confusion_matrix(y_test, pred)


# In[38]:


model.save("NN.h5")


# In[40]:


model.save_weights("NN_Weight.h5")


# ### Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


log_reg = LogisticRegression(random_state=10)


# In[ ]:


log_reg.fit(X_train, y_train)
pred = log_reg.predict(X_test)


# In[ ]:


accuracy_score(y_test, pred)


# ### Ensemble All

# In[41]:


from sklearn.ensemble import VotingClassifier


# In[45]:


voting_clf = VotingClassifier(
estimators=[('lr', log_reg), ('rf', forest),('dt', tree)], voting='hard')
voting_clf.fit(X_train, y_train)


# In[46]:


pred = voting_clf.predict(X_test)


# In[47]:


accuracy_score(y_test, pred)


# In[48]:


confusion_matrix(y_test, pred)

