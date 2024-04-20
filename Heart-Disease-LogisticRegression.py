#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\pankaj singh mehta\Pictures\ml data\heart_cleveland_upload.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


plt.figure(figsize=(6,4))
sns.countplot(x='sex',data=df)
plt.title('Count by Gender') #male:1 & female:0
plt.show()


# In[9]:


for columns in df.columns:
    print( columns , len(df[columns].value_counts()))


# In[10]:


df.condition.value_counts() # 0-- healthy heart ,1-- defective heart


# In[11]:


X = df.drop(['condition'],axis=1)
Y= df['condition']


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y , train_size=0.80 , random_state=6)


# In[13]:


print(X.shape,X_train.shape,X_test.shape)


# In[14]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[15]:


lr.fit(X_train,Y_train)


# In[16]:


X_train_predicition = lr.predict(X_train) #prediction on treining data


# In[17]:


from sklearn.metrics import accuracy_score


# In[18]:


training_data_accuracy = accuracy_score( X_train_predicition ,Y_train )


# In[19]:


print('training_data_accuracy:',training_data_accuracy)


# In[20]:


X_test_predicition = lr.predict(X_test) #prediction on test data


# In[21]:


test_data_accuracy = accuracy_score( X_test_predicition ,Y_test )


# In[22]:


print('test_data_accuracy:',test_data_accuracy)


# In[ ]:




