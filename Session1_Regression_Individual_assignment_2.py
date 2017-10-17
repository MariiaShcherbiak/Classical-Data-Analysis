
# coding: utf-8

# # Subject: Classical Data Analysis
# 
# ## Session 1 - Regression
# 
# ### Individual assignment 2
# 
# Elaborate with only the first feature of the "iris" sklearn dataset, in order to illustrate a two-dimensional plot of this regression technique. 
# 
# Use the field “sepal width (cm)” as independent variable and the field “sepal length (cm)” as dependent variable.
# 
# Calculate the coefficients, the residual sum of squares and the variance score.
# 
# - Interpret and discuss the Regression Results.
# - Commit scripts in your GitHub account. You should export your solution code (.ipynb notebook) and push it to your repository “ClassicalDataAnalysis”.
# 
# The following are the tasks that should complete and synchronize with your repository “ClassicalDataAnalysis” until October 13. Please notice that none of these tasks is graded, however it’s important that you correctly understand and complete them in order to be sure that you won’t have problems with further assignments.

# # Import the Python Libraries

# In[2]:

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# # Load the iris dataset

# In[5]:

iris = datasets.load_iris()
print (iris.DESCR)


# # Use only one feature for the dependent variable (sepal length (cm))

# In[22]:

iris_y = iris.data[:, np.newaxis, 0]
iris_y


# # Split the data of the dependent variable into training/testing sets

# In[23]:

iris_y_train = iris_y[:-20]
iris_y_test = iris_y[-20:]
iris_y_train


# # Use only one feature for the independent variable (sepal width (cm))

# In[24]:

iris_X = iris.data[:, np.newaxis, 1]
iris_X


# # Split the data of the independent variable into training/testing sets

# In[25]:

iris_X_train = iris_X[:-20]
iris_X_test = iris_X[-20:]
iris_X_train


# # Create linear regression object

# In[26]:

regr = linear_model.LinearRegression()


# # Train the model using the training sets

# In[27]:

regr.fit(iris_X_train, iris_y_train)


# # Make predictions using the testing set

# In[30]:

iris_y_pred = regr.predict(iris_X_test)


# # The coefficients

# In[31]:

print('Coefficients: \n', regr.coef_)


# # The mean squared error

# In[32]:

print("Mean squared error: %.2f"
      % mean_squared_error(iris_y_test, iris_y_pred))


# # Explained variance score: 1 is perfect prediction

# In[33]:

print('Variance score: %.2f' % r2_score(iris_y_test, iris_y_pred))


# # Plot the Regression Line

# In[34]:

plt.scatter(iris_X_test, iris_y_test,  color='black')
plt.plot(iris_X_test, iris_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:



