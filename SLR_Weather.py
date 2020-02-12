#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("D:/DS_TriS/Weather.csv")


# In[4]:


data.head()


# 

# In[5]:


data.columns


# In[6]:


data.shape


# In[7]:


data.describe()


# In[10]:


data.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()


# In[11]:


import seaborn as seabornInstance


# In[16]:


plt.figure(figsize = (15,10))
plt.tight_layout()
seabornInstance.distplot(data['MaxTemp'])


# In[17]:


X = data['MinTemp'].values.reshape(-1,1)
Y = data['MaxTemp'].values.reshape(-1,1)


# In[18]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)


# In[20]:


regressor = LinearRegression()
regressor.fit(X_train,Y_train)        #Training the algorithm


# In[21]:


#intercept
regressor.intercept_


# In[22]:


#slop
regressor.coef_


# In[24]:


#Prediction
y_pred = regressor.predict(X_test)
y_pred


# In[27]:


#Comparision B/W Actual and Predicted values
df = pd.DataFrame({'Actual' : Y_test.flatten(), 'Predicted' : y_pred.flatten()})
df


# In[30]:


df[20:40]


# In[33]:


#First 30 records
df1 = df.head(30)
df1.plot(kind='bar',figsize = (15,10))
plt.grid(which='major',linestyle='-',color='green',linewidth = '0.5')
plt.grid(which='minor',linestyle=':',color='black',linewidth = '0.5')


# In[34]:


#plotting our straight line with test data
plt.scatter(X_test,Y_test,color='blue')
plt.plot(X_test,y_pred,color='red')


# In[35]:


#Mean_Absolute_Error(MSE)
metrics.mean_absolute_error(Y_test,y_pred)


# In[36]:


#Mean_Squqred_Error(MSE)
metrics.mean_squared_error(Y_test,y_pred)


# In[37]:


#Root_Mean_Squqred_Error
np.sqrt(metrics.mean_squared_error(Y_test,y_pred))


# In[43]:


np.mean(data['MinTemp']+data['MaxTemp'])/2


# In[44]:


#So our algo is not that perfect as we get more than 10% of our mean of all temprature


# In[ ]:




