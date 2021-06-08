#!/usr/bin/env python
# coding: utf-8

# In[534]:


#Linear Reg. using sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[535]:


data=pd.read_csv(r'C:\Users\blankblack\Desktop\data_for_lr.csv')


# In[536]:


data.head(10)


# In[537]:


data.info()


# In[552]:


#handle null value
data=data.dropna()


# In[553]:


data.info()


# In[554]:


data.shape


# In[555]:


#splitting data
training_input=np.array(data.x[0:500]).reshape(500,1)
training_output=np.array(data.y[0:500]).reshape(500,1)
test_input=np.array(data.x[500:700]).reshape(199,1)
test_output=np.array(data.y[500:700]).reshape(199,1)


# In[558]:


#linear regression
#1.training model
from sklearn.linear_model import LinearRegression

linear_regressor=LinearRegression()
linear_regressor.fit(train_input,train_output)


# In[557]:


linear_regressor


# In[560]:


#predict test input
predicted_value=linear_regressor.predict(test_input.reshape(-1,1))


# In[561]:


predicted_value


# In[562]:


test_output


# In[564]:


from sklearn.metrics import mean_squared_error

error=mean_squared_error(test_output,predicted_value)


# In[565]:


error


# In[568]:


#visualizing error
#original hypo
plt.plot(test_input,test_output,'*',color='green')

#model hypo
plt.plot(test_input,predicted_value,'_',color='red')
plt.xlabel('input')
plt.ylabel('output')
plt.show()


# In[ ]:




