#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\pankaj singh mehta\Desktop\DATA_SET\titanic.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.columns


# In[7]:


df.info()


# In[6]:


df.isnull().sum()


# In[9]:


round(df.isnull().sum()/len(df),2)*100


# In[10]:


for columns in df.columns:
    print(columns , len(df[columns].value_counts()))


# In[43]:


plt.figure(figsize=(6,3))
sns.countplot(x='Pclass',hue='Survived',data=df)
plt.show()


# In[48]:


plt.figure(figsize=(6,4))
sns.countplot(x='Sex',data=df)
plt.title('Count by Gender') #male:1 & female:0
plt.show()


# In[49]:


plt.figure(figsize=(6,3))
sns.countplot(x='Sex',hue='Survived',data=df)
plt.title('Survival Count by Gender') #male:1 & female:0
plt.show()


# In[45]:


plt.figure(figsize=(6,3))
sns.countplot(x='Embarked',hue='Survived',data=df)
plt.show()


# In[28]:


df['Fare'].max()


# In[17]:


df['Age'].max()


# In[11]:


df['Age'].fillna(df['Age'].mean(), inplace = True)


# In[12]:


df.isnull().sum()


# In[14]:


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)


# In[15]:


df.isnull().sum()


# In[32]:


plt.figure(figsize=(8,4))
sns.heatmap(df.corr(), cmap='rocket',annot=True)
plt.show()


# In[33]:


df.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'],axis=1,inplace = True)


# In[34]:


df.head()


# In[36]:


df.replace({'Sex':{'male':1,'female':0}},inplace = True)


# In[56]:


status = pd.get_dummies(df['Embarked'])
df = pd.concat([df,status],axis=1)


# In[57]:


df.head()


# In[58]:


df.drop(['Embarked'],axis=1,inplace = True)


# In[59]:


df.head()


# In[61]:


X =df.drop('Survived',axis=1)
y = df['Survived']


# In[62]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)


# In[63]:


print(X_train.shape , X_test.shape)


# In[64]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[66]:


y_pred = lr.predict(X_test)


# In[67]:


from sklearn.metrics import r2_score
r_squared= r2_score(y_test , y_pred)


# In[69]:


print(r_squared)


# In[ ]:




