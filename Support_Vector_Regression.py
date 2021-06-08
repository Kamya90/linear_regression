#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


# In[14]:


from sklearn.datasets import load_boston

boston_data=load_boston()
boston_data


# In[8]:


data=pd.DataFrame(boston_data.data, columns=boston_data.feature_names)


# In[9]:


boston_data.feature_names


# In[10]:


data.head()


# In[11]:


data['MEDV']=boston_data.target


# In[12]:


data.head()


# In[15]:


data.info()#less entries


# In[16]:


data.describe()


# In[17]:


data.isnull().sum()


# In[18]:


x=data.drop(['MEDV'], axis=1)
y=data['MEDV']


# In[19]:


x.head()


# In[20]:


y.head()


# In[21]:


#scaling
sc_x=StandardScaler()
x=sc_x.fit_transform(x)


# In[22]:


x#range -1 to 1


# In[23]:


#split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=0)


# In[24]:


xtrain.shape


# In[25]:


xtest.shape


# In[26]:


ytrain.shape


# In[29]:


ytest.shape


# In[30]:


#svr training odel
svr=SVR(kernel='rbf')#kernel selection hit and trial

svr.fit(xtrain,ytrain)


# In[32]:


#prediction
ypredict=svr.predict(xtest)


# In[34]:


ypredict


# In[35]:


ytest.head()


# In[38]:


#erroe
from sklearn.metrics import mean_squared_error
error=mean_squared_error(ypredict,ytest)


# In[39]:


error


# In[ ]:


#data not sufficient dataset too small

