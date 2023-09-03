#!/usr/bin/env python
# coding: utf-8

# In[31]:


df = pd.read_excel("cpu.xlsx")
df


# drop rows with missing values
df.dropna(inplace=True)

# save cleaned dataframe to a CSV file
df.to_csv('cleaned_cpu_data.csv', index=False)



# In[35]:


#cleaning the data
import pandas as pd
df= pd.read_csv("cleaned_cpu_data.csv")
df


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
df = pd.read_csv("cleaned_cpu_data.csv")

# Create separate dataframes for Intel and AMD processors
intel_df = df[df['processor_name'].str.contains("Intel")]
amd_df = df[df['processor_name'].str.contains("AMD")]


# In[37]:


intel_df


# In[38]:


amd_df


# In[39]:


#cleaning and saving the data for Intel
intel_df.to_csv('intel_data_clean.csv', index=False)


# In[40]:


#cleaning and saving the data for AMD
amd_df.to_csv('amd_data_clean.csv', index=False)


# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
df = pd.read_csv("intel_data_clean.csv")
Data_intel = df


# In[42]:


Data_intel


# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
df = pd.read_csv("amd_data_clean.csv")
Data_AMD = df


# In[44]:


Data_AMD


# In[73]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# Define the predictor variables (features)
X = df[['value', 'avg_bench', 'memory_pts', 'core_pts', 'mkt_share', 'age_months', 'price']]

# Define the target variable
y = df['user_rating']

# Add a constant term to the predictor variables
X = sm.add_constant(X)

# Create a linear regression model and fit it to the data
model = sm.OLS(y, X)
results = model.fit()

# Print the summary of the model
print(results.summary())


# explanation The OLS regression results show that the overall model has an R-squared value of 0.616,
#indicating that the model explains 61.6% of the variance in the target variable. 
#The F-statistic has a p-value of less than 0.05, indicating that the overall model is significant.
#Looking at the coefficients, we can see that only the "core_pts", "mkt_share", "age_months", and "price" 
#variables have a statistically significant impact on the target variable.

#The "core_pts" variable has a positive coefficient of 0.0147, indicating that as the core points increase, 
#the user rating tends to increase as well.

#The "mkt_share" variable has a positive coefficient of 6.8692, indicating that as the market share increases,
#the user rating tends to increase as well.

#The "age_months" variable has a negative coefficient of -0.0790, indicating that as the age of the processor 
#increases, the user rating tends to decrease.

#The "price" variable has a negative coefficient of -0.0469, indicating that as the price of the processor 
#increases, the user rating tends to decrease.

#The other variables, including "value", "avg_bench", and "memory_pts", 
#do not have statistically significant coefficients, indicating that they do not have a significant 
#impact on the user rating.


# In[72]:


import seaborn as sns

# Load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# Define the predictor variables (features)
X = df[['value', 'avg_bench', 'memory_pts', 'core_pts', 'mkt_share', 'age_months', 'price']]

# Define the target variable
y = df['user_rating']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Predict the target variable using the model
y_pred = model.predict(X)

# Plot the scatterplot with a regression line in red color
sns.regplot(x=X['core_pts'], y=y, color='blue', line_kws={'color': 'red'})

# Set the plot labels
plt.xlabel('core_pts')
plt.ylabel('User Rating')

# Show the plot
plt.show()


# In[90]:


import seaborn as sns

# Load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# Define the predictor variables (features)
X = df[['value', 'avg_bench', 'memory_pts', 'core_pts', 'mkt_share', 'age_months', 'price']]

# Define the target variable
y = df['user_rating']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Predict the target variable using the model
y_pred = model.predict(X)

# Plot the scatterplot with a regression line in red color
sns.regplot(x=X['mkt_share'], y=y, color='green', line_kws={'color': 'red'})

# Set the plot labels
plt.xlabel('mkt_share')
plt.ylabel('User Rating')

# Show the plot
plt.show()


# In[91]:


import seaborn as sns

# Load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# Define the predictor variables (features)
X = df[['value', 'avg_bench', 'memory_pts', 'core_pts', 'mkt_share', 'age_months', 'price']]

# Define the target variable
y = df['user_rating']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Predict the target variable using the model
y_pred = model.predict(X)

# Plot the scatterplot with a regression line in red color
sns.regplot(x=X['age_months'], y=y, color='purple', line_kws={'color': 'red'})

# Set the plot labels
plt.xlabel('age_months')
plt.ylabel('User Rating')

# Show the plot
plt.show()


# In[95]:


import seaborn as sns

# Load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# Define the predictor variables (features)
X = df[['value', 'avg_bench', 'memory_pts', 'core_pts', 'mkt_share', 'age_months', 'price']]

# Define the target variable
y = df['user_rating']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Predict the target variable using the model
y_pred = model.predict(X)

# Plot the scatterplot with a regression line in red color
sns.regplot(x=X['price'], y=y, color='orange', line_kws={'color': 'red'})

# Set the plot labels
plt.xlabel('price')
plt.ylabel('User Rating')

# Show the plot
plt.show()


# In[75]:


import pandas as pd
import statsmodels.api as sm

# Load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# Separate the data for Intel and AMD processors
df_intel = df[df['processor_name'].str.contains('Intel')]
df_amd = df[df['processor_name'].str.contains('AMD')]

# Define the predictor variables (features) for Intel processors
X_intel = df_intel[['value', 'avg_bench', 'memory_pts', 'core_pts', 'mkt_share', 'age_months', 'price']]

# Define the target variable for Intel processors
y_intel = df_intel['user_rating']

# Add a constant term to the predictor variables for Intel processors
X_intel = sm.add_constant(X_intel)

# Create a linear regression model and fit it to the data for Intel processors
model_intel = sm.OLS(y_intel, X_intel)
results_intel = model_intel.fit()

# Print the summary of the model for Intel processors
print('Intel Processor Model\n', results_intel.summary())

# Define the predictor variables (features) for AMD processors
X_amd = df_amd[['value', 'avg_bench', 'memory_pts', 'core_pts', 'mkt_share', 'age_months', 'price']]

# Define the target variable for AMD processors
y_amd = df_amd['user_rating']

# Add a constant term to the predictor variables for AMD processors
X_amd = sm.add_constant(X_amd)

# Create a linear regression model and fit it to the data for AMD processors
model_amd = sm.OLS(y_amd, X_amd)
results_amd = model_amd.fit()

# Print the summary of the model for AMD processors
print('AMD Processor Model\n', results_amd.summary())


# In[ ]:


#Based on the regression results, there are some differences between the AMD and Intel processor models. 
#The R-squared value for the Intel model is 0.63, indicating that the model explains 63% of the variance 
#in user ratings. The R-squared value for the AMD model is much higher, at 0.939, indicating that the model
#explains 93.9% of the variance in user ratings. This suggests that the AMD model may be a better fit for 
#the data than the Intel model.

#Looking at the coefficient values, there are also some differences between the two models. For example, 
#the coefficient for the "memory_pts" variable is positive for the Intel model and negative for the AMD model, 
#suggesting that this variable has a different effect on user ratings for the two processor types. The 
#coefficient for the "avg_bench" variable is also positive for the AMD model and not significant for the 
#Intel model, suggesting that this variable may be more important for predicting user ratings for AMD processors.

#It is important to note that the two models are not directly comparable, as they are fitted on different 
#sets of observations. Nevertheless, the results indicate that there are some differences between the two models,
#and that the AMD model may be a better fit for the data.







# In[79]:





# In[87]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# split the data into AMD and Intel processors
amd_df = df[df['processor_name'].str.contains('AMD')]
intel_df = df[df['processor_name'].str.contains('Intel')]

# perform two-sample t-test to compare mean core points of AMD and Intel processors
t_stat_core, p_value_core = ttest_ind(amd_df['core_pts'], intel_df['core_pts'], equal_var=False)

# perform two-sample t-test to compare mean mkt_share of AMD and Intel processors
t_stat_mkt, p_value_mkt = ttest_ind(amd_df['mkt_share'], intel_df['mkt_share'], equal_var=False)

# perform two-sample t-test to compare mean age_months of AMD and Intel processors
t_stat_age, p_value_age = ttest_ind(amd_df['age_months'], intel_df['age_months'], equal_var=False)

# perform two-sample t-test to compare mean price of AMD and Intel processors
t_stat_price, p_value_price = ttest_ind(amd_df['price'], intel_df['price'], equal_var=False)

# print the results

print('T-statistic(core_pts):', t_stat_core)
print('P-value(core_pts):', p_value_core)
print('T-statistic (mkt_share):', t_stat_mkt)
print('P-value (mkt_share):', p_value_mkt)
print('T-statistic (age_months):', t_stat_age)
print('P-value (age_months):', p_value_age)
print('T-statistic (price):', t_stat_price)
print('P-value (price):', p_value_price)


# In[88]:


#core_pts: The t-test comparing the mean core points of AMD and Intel processors resulted in a t-statistic of -1.03 
#and a p-value of 0.30. Since the p-value is greater than 0.05, we cannot reject the null hypothesis that the mean 
#core points of AMD and Intel processors are the same. Therefore, we can conclude that there is no significant difference 
#in the mean core points of AMD and Intel processors.

#mkt_share: The t-test comparing the mean market share of AMD and Intel processors resulted in a t-statistic of -2.47 
#and a p-value of 0.014. Since the p-value is less than 0.05, we can reject the null hypothesis that the mean market 
#share of AMD and Intel processors are the same. Therefore, we can conclude that there is a significant difference in 
#the mean market share of AMD and Intel processors.

#age_months: The t-test comparing the mean age in months of AMD and Intel processors resulted in a t-statistic of -0.66 
#and a p-value of 0.51. Since the p-value is greater than 0.05, we cannot reject the null hypothesis that the mean age in 
#months of AMD and Intel processors are the same. Therefore, we can conclude that there is no significant difference in 
#the mean age in months of AMD and Intel processors.

#price: The t-test comparing the mean price of AMD and Intel processors resulted in a t-statistic of 0.12 and a p-value 
#of 0.91. Since the p-value is greater than 0.05, we cannot reject the null hypothesis that the mean price of AMD and 
#Intel processors are the same. Therefore, we can conclude that there is no significant difference in the mean price of 
#AMD and Intel processors.


#We used the t-test to determine whether there is a statistically significant difference between the means of two samples, 
#in this case, the means of the 'core_pts', 'mkt_share', 'age_months', and 'price' variables for AMD and Intel processors.

#The t-test is a parametric test that assumes the data are normally distributed and the variances of the two groups being 
#compared are equal.


# In[ ]:





# In[89]:


import pandas as pd

# load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# describe the data
desc = df.describe()

# print the results
print(desc)


# In[94]:


import pandas as pd

# Load the data
df = pd.read_csv('cleaned_cpu_data.csv')

# Get the row with the highest core points for AMD processors
amd_best_core = df[df['processor_name'].str.contains('AMD')].sort_values('core_pts', ascending=False).iloc[0]

# Get the row with the highest core points for Intel processors
intel_best_core = df[df['processor_name'].str.contains('Intel')].sort_values('core_pts', ascending=False).iloc[0]

# Get the row with the highest market share for AMD processors
amd_best_share = df[df['processor_name'].str.contains('AMD')].sort_values('mkt_share', ascending=False).iloc[0]

# Get the row with the highest market share for Intel processors
intel_best_share = df[df['processor_name'].str.contains('Intel')].sort_values('mkt_share', ascending=False).iloc[0]

# Print the results
print('Best CPU for AMD based on core points:\n', amd_best_core)
print('\nBest CPU for Intel based on core points:\n', intel_best_core)
print('\nBest CPU for AMD based on market share:\n', amd_best_share)
print('\nBest CPU for Intel based on market share:\n', intel_best_share)

