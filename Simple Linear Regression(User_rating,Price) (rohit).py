#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data into a Pandas DataFrame
df = pd.read_excel('/Users/rohithvarma/Desktop/cpu.xlsx')

# Select only the columns of interest
df = df[['price', 'user_rating']]

# Remove rows with missing data
df = df.dropna()

# Plot the data points
plt.scatter(df['price'], df['user_rating'])
plt.title('CPU Price vs User Rating')
plt.xlabel('Price')
plt.ylabel('User Rating')

# Fit a linear regression model
model = LinearRegression().fit(df[['price']], df['user_rating'])

# Plot the linear regression line
plt.plot(df['price'], model.predict(df[['price']]), color='red')

# Show the plot
plt.show()

#    The slope of the line indicates the rate of change in user ratings per unit increase in CPU price.
#    A positive slope indicates that as the CPU price increases, the user rating also tends to increase, 
#    while a negative slope indicates the opposite. In this case, since the slope is positive 
#    (as indicated by the equation of the line on the plot), we can conclude that there is a positive 
#    relationship between CPU price and user rating, i.e., as CPU price increases, user rating 
#    tends to increase as well.


# In[ ]:




