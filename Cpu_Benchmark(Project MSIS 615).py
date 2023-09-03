#!/usr/bin/env python
# coding: utf-8

# In[10]:


#The code for the project webcrawling 

import sqlite3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# Constants
WEBDRIVER_PATH = "C:/webdrivers/chromedriver.exe"
BASE_URL = "https://cpu.userbenchmark.com/"
START_PAGE = 1
END_PAGE = 10

# Set up Selenium driver
service = Service(WEBDRIVER_PATH)
driver = webdriver.Chrome(service=service)

# Set up database connection and cursor
conn = sqlite3.connect("user_cpu.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS processors
                (id INTEGER PRIMARY KEY,
                processor_name TEXT,
                user_rating TEXT,
                value TEXT,
                avg_bench TEXT,
                memory_pts TEXT,
                core_pts TEXT,
                mkt_share TEXT,
                age_months TEXT,
                price TEXT)''')

# Loop through pages
for page_number in range(START_PAGE, END_PAGE+1):
    # Construct page URL
    page_url = f"{BASE_URL}/Explore/MarketShareNvidia/{page_number}"
    driver.get(page_url)

    # Find table rows and extract data
    table_rows = driver.find_elements("xpath", '/html/body/div[2]/div/div[6]/form/div[2]/table/tbody/tr')

    for row in table_rows:
        row_number = row.find_element("xpath", "./td[1]/div").text
        processor_name = row.find_element("xpath", "./td[2]/div/div[2]/span").text.replace("Compare\n", "")
        user_rating = row.find_element("xpath", "./td[3]/div[1]").text.replace("▲\n", "").replace("\n▼", "")
        value = row.find_element("xpath", "./td[4]").text[:-6]
        avg_bench = row.find_element("xpath", "./td[5]/div[1]/div").text[-3:]
        memory_pts = row.find_element("xpath", "./td[6]/div[1]").text
        core_pts = row.find_element("xpath", "./td[7]/div[1]").text
        mkt_share = row.find_element("xpath", "./td[8]/div[1]").text
        age_months = row.find_element("xpath", "./td[9]/div[1]").text.replace("+", "")
        price = row.find_element("xpath", "./td[10]").text[:-4]

        # Insert data into table
        cursor.execute('''INSERT INTO processors (processor_name, user_rating, value, avg_bench, memory_pts, core_pts, mkt_share, age_months, price)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (processor_name, user_rating, value, avg_bench, memory_pts, core_pts, mkt_share, age_months, price))
        conn.commit()

# Close database connection and Selenium driver
cursor.close()
conn.close()
driver.quit()


# In[1]:


# exporting the data into excel to check the data
import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect("user_cpu.db")

# Query data from database
data = pd.read_sql_query("SELECT * FROM processors", conn)

# Export data to Excel file
data.to_excel("cpu_data.xlsx", index=False)

# Close database connection
conn.close()


# In[2]:


import pandas as pd

# Read Excel file into a Pandas DataFrame
df = pd.read_excel("cpu_data.xlsx")
# to see the data frame and see if we need to clean the data
df


# In[3]:


#cleaning the data
import pandas as pd

# Read Excel file into a Pandas DataFrame
df = pd.read_excel("cpu_data.xlsx")

# Replace \n and $ characters from price column
df["price"] = df["price"].str.replace("\n", "").str.replace("$", "")

# Export the DataFrame to a new Excel file
df.to_excel("cpu.xlsx", index=False)

# Print confirmation message
print("Data exported to cpu.xlsx.")


# In[3]:


df


# In[4]:


# drop rows with missing values
df.dropna(inplace=True)

# save cleaned dataframe to a CSV file
df.to_csv('cleaned_cpu_data.csv', index=False)


# In[5]:


#cleaning the data
import pandas as pd
df= pd.read_csv("cleaned_cpu_data.csv")


# In[6]:


df


# In[ ]:




