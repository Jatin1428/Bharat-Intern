#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten


# In[2]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[3]:


x_train.shape


# In[4]:


y_train


# In[5]:


import matplotlib.pyplot as plt
plt.imshow(x_train[5])


# In[6]:


x_train[0]


# In[7]:


model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation="relu"))
model.add(Dense(30,activation="relu"))
model.add(Dense(10,activation="softmax"))


# In[8]:


model.summary()


# In[9]:


model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam",metrics="accuracy")


# In[10]:


data=model.fit(x_train,y_train,epochs=30,validation_split=0.2)


# In[15]:


y_prob = model.predict(x_test)


# In[16]:


y_pred=y_prob.argmax(axis=1)


# In[17]:


y_pred


# In[18]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)


# In[19]:


accuracy


# In[13]:


plt.plot(data.history['loss'])
plt.plot(data.history['val_loss'])


# In[12]:


plt.plot(data.history['accuracy'])
plt.plot(data.history['val_accuracy'])


# In[11]:


plt.imshow(x_test[0])


# In[20]:


value=model.predict(x_test[0].reshape(1,28,28)).argmax(axis=1)


# In[21]:


print(value)


# In[22]:


plt.imshow(x_test[11])


# In[23]:


expected_value=model.predict(x_test[11].reshape(1,28,28)).argmax(axis=1)
print(expected_value)


# In[ ]:




