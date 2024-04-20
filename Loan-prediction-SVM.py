#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv(r"C:\Users\pankaj singh mehta\Pictures\ml data\loan_prediction.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.isnull().sum()/len(df)*100


# In[9]:


round(df.isnull().sum()/len(df)*100)


# In[10]:


for columns in df.columns:
    print(columns , len(df[columns].value_counts()))


# In[11]:


plt.figure(figsize=(6,3))
sns.countplot(x='Gender',hue='Loan_Status',data=df)
plt.title('Count by Gender loan given')
plt.show()


# In[12]:


plt.figure(figsize=(6,3))
sns.countplot(x='Married', hue='Loan_Status', data= df)
plt.title('count of married people get loan')
plt.show()


# In[13]:


plt.figure(figsize=(6,3))
sns.countplot(x='Education', hue='Loan_Status', data= df)
plt.title('count of people getting loan by education')
plt.show()


# In[14]:


plt.figure(figsize=(6,3))
sns.countplot(x='Self_Employed', hue='Loan_Status', data= df)
plt.title('count of Self_Employed  people get loan')
plt.show()


# In[15]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)  #filling the null value


# In[16]:


df['Dependents'].fillna(df['Dependents'].mode()[0], inplace = True)


# In[17]:


df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace = True)


# In[18]:


df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)


# In[19]:


df = df.dropna()


# In[20]:


df.shape


# In[21]:


varlist = ['Married','Self_Employed']  #binary_mapping

def binary_map(x):
    return x.map({'Yes':1,'No':0})
df[varlist]= df[varlist].apply(binary_map)


# In[22]:


# convert categorical data to numerical value
df.replace({'Gender':{'Male':1,'Female':0},'Education':{'Graduate':1,'Not Graduate':0},'Loan_Status':{'Y':1,'N':0}},inplace=True)


# In[23]:


df.head()


# In[24]:


df['Dependents'].value_counts()


# In[25]:


df = df.replace(to_replace='3+',value=4)


# In[26]:


status = pd.get_dummies(df['Property_Area'])


# In[27]:


df= pd.concat([df, status], axis=1)
df.head()


# In[28]:


df.drop(['Loan_ID','Property_Area'],axis=1,inplace= True)


# In[30]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), cmap='rocket',annot=True)
plt.show()


# In[31]:


x = df.drop(['Loan_Status'],axis=1)
y = df['Loan_Status']


# In[32]:


x.shape


# In[33]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20 ,random_state=5) #train_test_split


# In[34]:


print(x_train.shape,x_test.shape)


# In[35]:


# training the model
svc = svm.SVC(kernel='linear')


# In[36]:


svc.fit(x_train,y_train)


# In[37]:


# model evaluation
# accuracy score on training data
x_train_prediction = svc.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


# In[39]:


print('Accuracy on training data:',training_data_accuracy)


# In[40]:


# accuracy score in test data
x_test_prediction = svc.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[41]:


print('Accuracy on training data:',test_data_accuracy)


# In[ ]:




