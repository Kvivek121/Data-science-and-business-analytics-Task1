#!/usr/bin/env python
# coding: utf-8

# # Author: Vivek kaushik
# 
# 
# 

# ## 1. Importing all libraries required in this notebook

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2.Reading data from remote link

# In[2]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data


# ## 3.Plotting the distribution of scores

# In[3]:



s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# ## 4.Preparing the data
# 
#  Dividing the data into "attributes" (inputs) and "labels" (outputs).

# In[4]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# Spliting the data into training and test sets.

# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[6]:


X_train


# In[7]:


X_test


# In[8]:


y_train


# In[9]:


y_test


# ### **5.Training the Algorithm**
# We have split our data into training and testing sets, and now is finally the time to train our algorithm. 

# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# ## 6.Plotting the regression line

# In[11]:


line = regressor.coef_*X+regressor.intercept_


# ## 7.Plotting for the test data
# 

# In[12]:


plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### **8.Making Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# ## 9.Testing data - In Hours

# In[13]:


print(X_test) 


# ## 10.Predicting the scores

# In[14]:


y_pred = regressor.predict(X_test) 


# In[15]:


y_pred


# ## 11.Comparing Actual vs Predicted

# In[16]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[17]:


hours = 9.25
own_pred = regressor.predict(np.array(hours).reshape(-1,1))
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### **12.Evaluating the model**
# 
# Evaluating the performance of algorithm. Here, we have chosen the mean square error.

# In[19]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# # Predicted Score = 93.69173248737538

# # Result
# ### The predicted score if a student studies for 9.25 hours/day is 93.69173248737538 with Mean absolute error of 4.183859899002975

# # Thank you
