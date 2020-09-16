#!/usr/bin/env python
# coding: utf-8

# In[1]:


# install the libraries

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')


# In[32]:


# load the data
df = pd.read_csv(r'C:\Users\akash\Downloads\SPY (2).csv')
df.head()


# In[33]:


# getting the number of trading days
df.shape


# In[34]:


# visualise the close price data
plt.figure(figsize=(18,8))
plt.title('SPY')
plt.xlabel('Days')
plt.ylabel('Close Price in USD')
plt.plot(df['Close'])
plt.show()


# In[35]:


# Getting the close price
df = df[['Close']]
print(df.head)


# In[36]:


# # Variable to predict "x" days out in the future:
# future_days = 25
# # create a new column(target data)
# df['Prediction'] = df[['Close']].shift(future_days)
# df.head()


# In[37]:


future_days = 10
df['Prediction'] = df[['Close']].shift(-future_days)
df.tail(5)


# In[38]:


# Create a feature Data set (x) and convert it to numpy array and remove the last 'x' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(X)


# In[39]:


# Create the target data set(y) and convert it to numpy array and get all of the target values except the last 'x' rows/days
Y = np.array(df['Prediction'])[:-future_days]
print(Y)


# In[40]:


# Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25)


# In[41]:


# Create the models
# create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train,y_train)
# Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)


# In[42]:


# Getting the last 'x' rows of the feature data set
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# In[43]:


# Show the model tree prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
#show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)


# In[44]:


#Visualize the data
predictions = tree_prediction

valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize= (18,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD($)')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])
plt.show()


# In[ ]:




