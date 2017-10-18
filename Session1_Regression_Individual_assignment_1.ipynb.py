
# coding: utf-8

# # Subject: Classical Data Analysis
# 
# ## Session 1 - Regression
# 
# ### Individual assignment 1
# 
# Develop a regression analysis in Statmodels (with and without a constant) and SKLearn, based on the Iris sklearn dataset. This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length.
# 
# See here for more information on this dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set 
# 
# Use the field “sepal width (cm)” as independent variable and the field “sepal length (cm)” as dependent variable.
# 
# - Interpret and discuss the OLS Regression Results.
# - Commit scripts in your GitHub account. You should export your solution code (.ipynb notebook) and push it to your repository “ClassicalDataAnalysis”.
# 
# The following are the tasks that should complete and synchronize with your repository “ClassicalDataAnalysis” until October 13. Please notice that none of these tasks is graded, however it’s important that you correctly understand and complete them in order to be sure that you won’t have problems with further assignments.

# # Linear Regression in Statsmodels

# ## Load the iris dataset

# In[9]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
iris = datasets.load_iris()
print (iris.DESCR)
from pandas.core import datetools

df = pd.DataFrame(iris.data, columns=iris.feature_names) 
df


# ### Regression model with Statsmodels and without a constant:

# In[12]:

df.head()
import statsmodels.api as sm
target = pd.DataFrame(iris.target, columns=["sepal length (cm)"]) 
X = df["sepal width (cm)"]
y = target["sepal length (cm)"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X) 
model.summary()


# ### Interpreting the Table 

# 

# ### Regression model with Statsmodels and with a constant:

# In[15]:

import statsmodels.api as sm 
X = df["sepal width (cm)"]
y = target["sepal length (cm)"]
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()


# ### Interpreting the Table 

# 

# # Linear Regression in SKLearn 

# In[16]:

from sklearn import linear_model
from sklearn import datasets 
iris = datasets.load_iris() 
df = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[18]:

df


# In[31]:

df2 = pd.DataFrame(df, columns=["sepal width (cm)"])


# In[ ]:




# In[32]:

target = pd.DataFrame(iris.target, columns=["sepal length (cm)"])


# In[33]:

X = df2
y = target["sepal length (cm)"]


# In[34]:

lm = linear_model.LinearRegression()
model = lm.fit(X,y)


# In[35]:

type(predictions)


# In[36]:

predictions = lm.predict(X)
print(predictions[0:5,])


# In[37]:

print(predictions)


# In[38]:

lm.score(X,y) 


# In[39]:

lm.coef_ 


# In[40]:

lm.intercept_ 


# In[ ]:



