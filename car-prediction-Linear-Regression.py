#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[6]:


df = pd.read_csv(r"C:\Users\pankaj singh mehta\Pictures\ml data\car data.csv")


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


for columns in df.columns:
    print(columns , len(df[columns].value_counts()))


# In[18]:


print(df['Fuel_Type'].value_counts())
print(df['Seller_Type'].value_counts())
print(df['Transmission'].value_counts())


# In[20]:


df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2},'Seller_Type':{'Dealer':0,'Individual':1},'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[22]:


df.info()


# In[21]:


df.head()


# In[25]:


plt.figure(figsize=(10,6))
sns.countplot(x='Car_Name',data=df)
plt.title('count of car by its name')
plt.show()


# In[26]:


X = df.drop(['Car_Name','Selling_Price'],axis=1)
Y = df['Selling_Price']


# In[27]:


print(Y)


# In[29]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=5)


# In[32]:


print(X_train.shape,X_test.shape)


# In[34]:


lr = LinearRegression()


# In[35]:


lr.fit(X_train,Y_train)


# In[36]:


#prediction on training data
training_data_prediction = lr.predict(X_train)


# In[37]:


# R squared Error
error_score = metrics.r2_score(Y_train,training_data_prediction)
print('r squared error:',error_score)


# In[38]:


# visualisation of actual price and pedicted price
plt.scatter(Y_train , training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual and predicted price')
plt.show()


# In[39]:


#prediction on training data
test_data_prediction = lr.predict(X_test)


# In[40]:


# R squared Error
error_score = metrics.r2_score(Y_test,test_data_prediction)
print('r squared error:',error_score)


# In[41]:


plt.scatter(Y_test , test_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual and predicted price')
plt.show()


# In[ ]:




