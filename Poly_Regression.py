#!/usr/bin/env python
# coding: utf-8

# In[1]:


#polynomial linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.axes as ax


# In[ ]:


data=pd.read_csv(r'C:\Users\blankblack\Desktop\data_for_lr.csv')


# In[ ]:


data.head(10)


# In[ ]:


data.info()


# In[ ]:


data=data.dropna()


# In[ ]:


data.info()


# In[ ]:


training_input=np.array(data.x[0:500]).reshape(500,1)
training_output=np.array(data.y[0:500]).reshape(500,1)
test_input=np.array(data.x[500:700]).reshape(199,1)
test_output=np.array(data.y[500:700]).reshape(199,1)

print("Training Input Shape = {}".format(training_input.shape))
print("Training Output Shape = {}".format(training_output.shape))
print("Testing Input Shape = {}".format(test_input.shape))
print("Testing Output Shape = {}".format(test_output.shape))


# In[ ]:


#linear regression
#training model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_regressor=PolynomialFeatures(degree=2)
train_input_poly=poly_regressor.fit_transform(train_input)
poly_regressor.fit(train_input,train_output)


# In[ ]:


train_input_poly


# In[ ]:


train_input_poly.shape


# In[ ]:


linear_regressor=LinearRegression()
linear_regressor.fit(train_input_poly,train_output)


# In[ ]:


#predicting test input
predicted_value=linear_regressor.predict(poly_regressor.fit_transform(test_input))


# In[ ]:


from sklearn.metrics import mean_squared_error
error=mean_squared_error(predicted_value, test_output)


# In[ ]:


error


# In[ ]:


#visualization
plt.plot(test_input,test_output,'_',color='green')

#model hypo
plt.plot(test_input,predicted_value,'.',color='red')
plt.title('poly_plot')
plt.xlabel('input')
plt.ylabel('output')
plt.show()

