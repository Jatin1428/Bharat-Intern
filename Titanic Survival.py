#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


td = pd.read_csv('Titanic_ds.csv')
td


# In[26]:


td['Age'].fillna(td['Age'].mean(),inplace = True)
td


# In[27]:


td.info()


# In[28]:


td['Sex'] = td['Sex'].map({'male': 1, 'female': 0})
td['Sex'].value_counts()


# In[29]:


sns.boxplot(x='Sex', y='Age', data= td) #male:1,female:0


# In[30]:


sns.histplot(td['Fare'],color='r')


# In[31]:


sns.countplot(x='Sex', data=td)


# In[32]:


sns.histplot(td['Age'])


# In[33]:


sns.countplot(x='Pclass',data=td,hue='Survived')
labels = {0: 'Not Survived', 1: 'Survived'}
plt.legend(title='Survived', labels=[f"{key} = {value}" for key, value in labels.items()])
plt.title('diffrent Pclass by Survival Status')
plt.show()


# In[34]:


sns.heatmap(td.corr(),annot=True)


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[36]:


X = td[['Age', 'Pclass', 'Sex']]
y = td['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)


# In[38]:


accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# In[42]:


sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




