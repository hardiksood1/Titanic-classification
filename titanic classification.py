#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


titanic_data = pd.read_csv(r"C:\Users\hp\Desktop\titanic.csv")


# In[4]:


titanic_data.head()


# In[10]:


titanic_data.describe()


# In[12]:


titanic_data.isnull().sum()


# In[13]:


titanic_data.shape


# In[ ]:





# In[5]:


titanic_data = pd.get_dummies(titanic_data, columns=['Sex'])
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[6]:


features = ['Pclass', 'Age', 'Sex_female', 'Sex_male']
X = titanic_data[features]
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[8]:


predictions = model.predict(X_test)


# In[9]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[ ]:




