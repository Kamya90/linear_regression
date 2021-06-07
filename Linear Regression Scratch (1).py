#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax


# In[2]:


data = pd.read_csv(r'C:\Users\blankblack\Desktop\data_for_lr.csv')


# In[3]:


data.head(10)


# In[4]:


data.info()


# #### Handling NULL Value

# In[5]:


data = data.dropna()
print("Shape of the dataset = {}".format(data.shape))


# In[6]:


data.info()


# #### Splitting data

# In[7]:


# training dataset and labels
train_input = np.array(data.x[0:500]).reshape(500, 1)
train_output = np.array(data.y[0:500]).reshape(500, 1)

test_input = np.array(data.x[500:700]).reshape(199, 1)
test_output = np.array(data.y[500:700]).reshape(199, 1)


# valid dataset and labels

# print the shapes
print("Train Input Shape = {}".format(train_output.shape))
print("Train Output  Shape = {}".format(train_output.shape))
print("Test Input Shape = {}".format(test_input.shape))
print("Test Output  Shape = {}".format(test_output.shape))


# In[ ]:


#f(x) = mx + c


# In[8]:


def forward_propagation(train_input, parameters):
    m = parameters['m']
    c = parameters['c']
    
    predictions = np.multiply(m, train_input) + c
    
    return predictions


# In[9]:


def cost_function(predictions, train_output):
    cost = np.mean((train_output - predictions)**2) * 0.5
    
    return cost


# In[10]:


def backward_propagation(train_input, train_output, predictions):
    derivatives = dict()
    df = (predictions - train_output)
    dm = np.mean(np.multiply(df, train_input))
    dc = np.mean(df)
    
    
    derivatives['dm'] = dm
    derivatives['dc'] = dc
    
    return derivatives


# In[11]:


def update_parameters(parameters, derivatives, learning_rate):
    parameters['m'] = parameters['m'] - learning_rate * derivatives['dm']
    parameters['c'] = parameters['c'] - learning_rate * derivatives['dc']
    
    return parameters


# In[ ]:



        


# In[23]:


#training the model
def train(train_inputs,train_output,learning_rate,iters):
    
    parameters=dict()
    parameters['m']=np.random.uniform(0,1)*0.10
    parameters['c']=np.random.uniform(0,1)*0.10
    
    plt.figure()
    loss=list()
    for i in range(iters):
        
        predictions=forward_propagation(train_inputs, parameters)
        
        cost=cost_function(predictions,train_output)
        
        loss.append(cost)
        print("Iteration = {}, Loss = {}".format(i+1, cost))
        
        fig, ax=plt.subplots()
        
        ax.plot(train_input,train_output,'+',label='original')#actual h
        ax.plot(train_input,predictions,'*',label='training')
        
        legend=ax.legend()
        plt.plot(train_input,train_output,'+')
        plt.plot(train_input,predictions,'*')
        plt.show()
        
        derivatives=backward_propagation(train_input,train_output,predictions)
        
        parameters=update_parameters(parameters, derivatives, learning_rate)
        
    return parameters,loss      


# In[24]:


#parameters1, loss1 = train(train_input, train_output, 0.01, 50)

parameters, loss = train(train_input, train_output, 0.0001, 50)


# In[25]:


print(parameters)


# In[26]:


test_predictions = test_input * parameters["m"] + parameters["c"]
plt.figure()
plt.plot(test_input, test_output, '+')
plt.plot(test_input, test_predictions, '.')
plt.show()


# In[17]:


cost_function(test_predictions, test_output)


# In[18]:


np.random.uniform(0,1)


# In[19]:


test_output


# In[20]:


test_predictions


# In[ ]:





# In[ ]:





# In[ ]:




