#!/usr/bin/env python
# coding: utf-8

# Samuel Castro Project Milestone 1 

# Project Subject Area: My project is going to consist of looking at COVID-19 data. I will comb, and clean the data using the methods that we will learn each week in this class.

# Data Sources:
#   
#       Flat File: CSV File
#        Description: This is a file that shows confirmed cases, deaths, recovered per country.
#            Link: https://www.kaggle.com/datasets/imdevskp/corona-virus-report
#    
#       API:
#        Description: This is an API designed for developers and machines to grab data about covid-19 quickly. You can 
#        get information on deaths,
#        vaccinated, recovered and so much more.
#            Link:https://github.com/M-Media-Group/Covid-19-API
#            
#       Website:
#        Description: This is a dataset that comes from wikipedia. This data shows the reported cases and deaths by   
#        country and territory.
#            Link: https://en.wikipedia.org/wiki/COVID-19_pandemic_by_country_and_territory

#     Relationships:
# 
# Each file is related to the other because they each show cases and deaths in each of there unique ways. For example, the wikepedia website shows territories while the other datasets do not.

# 250 Words describing how you plan to tackle the project, what the data means, ethical implications of your project scenario/topic, and what challenges you might face

# I plan to format each dataset into a more readable manner. I will narrow this down to only the more important columns such as deaths, confirmed cases, vaccinated, recovered. When I look at each dataset, the data will be scattered and I hope to clean it and remove the outliers and duplicates. The data is about covid and how covid affected the world and at what point it affected it the most. I believe that this touches base with the whole world so I believe this will have an emotional impact on someone but overall, this project will not have many ethical implications. I believe some of the challenges I will face is choosing which methods of transformation I will use. I fear that I am going to struggle to transform the data. I feel like there is not much else to do with the data because it is very self-explanatory. The data on COVID is extenisve and can range from a variety of different topics. I think that I am going to have to narrow my columns down so I can create a very readable and interpretive dataset. This project is about the data wrangling aspect and not the analysis so I do not want to get hung up on the analysis side of thinking. I believe this can be another challenge I can potentially face. I tend to think too much and get hung up on one idea. I want to learn how to truly data wrangle and data prep but my mind has been focused on analysis the past 2 quarters. I hope to change this perspective on this project.
# 

# What you believe you will have to do to the data to accomplish all 5 milestones and what your interpretation is of what the data means (you could provide a data dictionary or a summary of what the data is) – should be at least 250 words

# In order to accomplish all 5 milestones for this project I must make a connection beyond the death toll of COVID-19. I believe that in order to do this, I am going to need to clean and format the data in a specific manner. I hope to see when COVID impacted the world the most. To do this I must choose and organize the data in a smart manner. Viewing the impact of COVID is interesting to me because it can show me when the peak of the pandemic was and how it truly affected the world. The data is more than death tolls and vaccination records. It is a book that shows how impacted the world was. This data is the reason that schools, super markets, and economies were affected in tremendous ways. From the organized data in the end we will be able to create line graphs showing the peak of the pandemic and the lowest point. I will also be able to show which states were affected the most. The data set that interests me the most is the API data set because I have not worked with API's very much and this API gives the most information on COVID than the other 2 data sets. I hope to extract information from the API to combine with the other data sets and create very coherent final data set in SQL.

# Samuel Castro Project Milestone 2 

# In[1]:


import warnings 
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd


# In[2]:


df = pd.DataFrame(pd.read_csv("Desktop/DSC540/country_wise_latest.csv")) 
df.head()


# In[3]:


columns = list(df.columns) 
columns


# In[4]:


# Examining missing values
# Since there are no missing values we do not need to work
# on replacing or removing the missing values.
print("Missing values distribution: ") 
print(df.isnull().mean())
print("")


# In[5]:


# 1. The first change I will officially make to my dataset is to
#    remove columns that I do not find needed for my end product.
cols_dropped = ['Confirmed last week','1 week change', '1 week % increase','WHO Region']
df.drop(cols_dropped, inplace=True,axis=1)


# In[6]:


# Here I am checking the data type to get a better understanding of
# the variables. I will not change the data types because they are
# the correct data types.
print(df.dtypes)


# In[7]:


# 2. Here is the second change I will make to the data set. I will
#    simply be changing some column names to make it more sensible.
#    Some of the letters were not capitalized and I added cases to
#    a couple of headings.
new_names = {'Confirmed': 'Confirmed Cases', 'Recovered': 'Recovered Cases',
              'Active': 'Active Cases',
             'New cases':'New Cases',
             'New deaths':'New Deaths',
             'New recovered':'New Recovered'}
df.rename(columns=new_names, inplace=True)


# In[8]:


# Now I am going to check if any of this data is duplicated. Since
# there is no duplication then we can move on.
df.duplicated()


# In[9]:


# In this step we are going to check for outliers. We can clearly
# see that in deaths/100 recovered that infinity is not a
# realistic option. Every other value is logical for what the
# numbers represent.
df.describe()


# In[10]:


# We can then continue to analyze the column deaths/100 recovered
print(df['Deaths / 100 Recovered'].skew()) 
df['Deaths / 100 Recovered'].describe()


# In[11]:


# 3. Here is my third change to the dataset.
# I replaced infintiy with 0.
df.replace([np.inf], 0, inplace=True)


# In[12]:


# Now that we replaced infinity with 0 the values for the column
# are logical.
df['Deaths / 100 Recovered'].describe()


# In[13]:


# I am checking the output just in case the output does not make
# sense.
df.loc[180]


# In[14]:


# 4. The fourth change I made was to remove the whitspaces from the
# country.region column.
df['Country/Region'] = df['Country/Region'].str.strip()


# In[15]:


# 5. For my final change in the data I converted all numerical values to float types to make # it consistent across the board.
df = df.astype({'Confirmed Cases':'float',
                'Deaths':'float',
                'Recovered Cases':'float',
                'Active Cases': 'float',
                'New Cases': 'float',
                'New Recovered': 'float',
               'New Deaths':'float'})


# In[16]:


print(df.dtypes)


# In[17]:


# Updated and cleaned data head.
df


# In[18]:


# Save DataFrame as a new CSV file on the desktop
df.to_csv('/Users/captainsammy00/Desktop/CSV_Data.csv', index=False)


# Overall my project does not have many ethical implications and the cleaning process is very smooth because it was already fairly clean. Ethical implications may be involved in the future when I dive into the API because there is more information in that data. I went through the process of standard data cleaning. I checked for missing and duplicated values but the data set did not have any. I then looked for outliers and found out that infinity was used so I set out to replace all of these values with 0. This seemed to fix the problem and gave me normal results.

# Samuel Castro Project Milestone 3

# In[19]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
from IPython.display import display

# Fetch the Wikipedia page containing the tables

url = 'https://en.wikipedia.org/wiki/COVID-19_pandemic_by_country_and_territory#Total_cases,_deaths,_and_death_rates_by_count'
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content

soup = BeautifulSoup(response.text, 'html.parser') 

# Find all tables on the page

tables = soup.find_all('table', {'class': 'wikitable'})

# Process and display each table separately

for i, table in enumerate(tables, start=1):
    # Extract the table data and convert it into a pandas DataFrame 
    data = []
    for row in table.find_all('tr'):
        cols = row.find_all(['th', 'td']) 
        if cols:
            data.append([col.text.strip() for col in cols]) 
            
    df2 = pd.DataFrame(data)
            
    # Clean up and process the DataFrame
    
    df2.dropna(inplace=True)
    df2.dropna(axis=1, how='all', inplace=True)
    
    # Set the first row as the column headers
    df2.columns = df2.iloc[0] 
    df2 = df2[1:]
    
    # Display the table
    
    print(f"Table {i}:")
    display(df2)
    print()


# In[20]:


# I chose table 12 because this is the table that most correlates with my other data sources
# Convert Table 12 into a pandas DataFrame

df_table_12 = pd.read_html(str(tables[11]))[0] 
df_table_12.head()


# In[21]:


# Cleaning Step 1: Replace '—' with NaN and Handling missing values

df_table_12 = df_table_12.replace('—', np.nan)

df_table_12.fillna(0, inplace=True) 


# In[22]:


df_table_12.head()


# In[23]:


# Cleaning Step 2: Renaming columns

df_table_12 = df_table_12.rename(columns={"Country":"Country/Region",
                        "Deaths / million": "Deaths Per Million", 
                        "Deaths": "Total Deaths", 
                        "Cases": "Total Cases"})


# In[24]:


# Cleaning Step 3:Remove the '+' symbol and convert percentage columns to numeric if present

if '% of world population' in df_table_12.columns:
    df_table_12['% of world population'] = df_table_15['% of world population'].str.replace
    ('\+', '').str.rstrip('%').astype(float) / 100
df_table_12.tail() 


# In[25]:


# Cleaning Step 4:Drop specific rows and columnns that are not helpful 

cols_dropped = ['Unnamed: 0']
df_table_12.drop(cols_dropped, inplace=True,axis=1)

df_table_12.drop(239,inplace=True)
df_table_12.drop(238,inplace=True)
df_table_12.drop(0,inplace=True)
df_table_12.tail() 


# In[26]:


# Cleaning step 5: Converting into float and adding new column'New Cases' calculated as the difference 
# between 'Total Cases' and 'Total Deaths'

# Converting data types
df_table_12["Deaths Per Million"] = df_table_12["Deaths Per Million"].astype(float)
df_table_12["Total Deaths"] = df_table_12["Total Deaths"].astype(float)
df_table_12["Total Cases"] = df_table_12["Total Cases"].astype(float)

# Adding a new column for "New Cases"
df_table_12["New Cases"] = df_table_12["Total Cases"] - df_table_12["Total Deaths"]
df_table_12.head()


# In[27]:


print(df_table_12.dtypes)


# In[28]:


# Convert DataFrame to CSV file
df_table_12.to_csv('website_data.csv', index=False)

# Read the CSV file
df2 = pd.read_csv('website_data.csv')

# Display the DataFrame
print(df2)


# In[29]:


# Save DataFrame as a new CSV file on the desktop
df2.to_csv('/Users/captainsammy00/Desktop/Website_Data.csv', index=False)


# My project still does not have many ethical implications and the cleaning process for the website was smooth after I was able to extract the tables from the website. Ethical implications may be involved in the future when I dive into the API because there is more information in that data. This was a smaller dataset than my CSV file so I did not do some steps that I conducted in the CSV project milestone. I mostly changed very superficial stuff because it was again, a very clean dataset that I am working with.

# Samuel Castro: Project Milestone 4

# In[30]:


import requests 
import pandas as pd 
import numpy as np

# API endpoint URL
# I had to choose another API because the original API I chose had shut down.

url = "https://disease.sh/v3/covid-19/countries"

# Send GET request to the API

response = requests.get(url)

# Check if the request was successful (status code 200) 

if response.status_code == 200:
    
    # Extract the JSON data from the response
    
    data = response.json()
    
# Create a DataFrame from the data 

    df3 = pd.DataFrame(data)
    
# Rename columns

    headers = ["updated","Country", "Country Info", "Cases", "Today Cases", "Deaths", "Today Deaths", "Recovered", "Today Recovered", "Active", "Critical", "Cases Per One Million",
                "Deaths Per One Million", "Tests", "Tests Per One Million", "Population",
                "Continent", "One Case Per People", "One Death Per People", "One Test Per People",
               "Active Per One Million", "Recovered Per One Million", "Critical Per One Million"]
    df3.columns = headers 
    
else:
    print("Error:", response.status_code)


# In[31]:


# 1. Removing columns I find unnecessary
df3 = df3.drop("updated", axis=1)
df3 = df3.drop("Country Info", axis=1)
df3 = df3.drop("One Case Per People", axis=1) 
df3 = df3.drop("One Death Per People", axis=1) 
df3 = df3.drop("One Test Per People", axis=1) 
df3 = df3.drop("Today Deaths", axis=1)
df3 = df3.drop("Today Recovered", axis=1)


# In[32]:


# 2. My second change is going to rename some columns to make it more readable.
new_names = {"Country": "Country/Region",
"Cases": "Total Cases",
        "Today Cases": "New Cases",
        "Deaths": "Total Deaths",
        "Recovered": "Total Recovered",
        "Active": "Active Cases",
        "Critical": "Critical Cases",
        "Cases Per One Million": "Cases Per Million",
        "Deaths Per One Million": "Deaths Per Million",
        "Tests": "Total Tests",
        "Tests Per One Million": "Tests Per Million",
        "Continent": "Continent",
        "Active Per One Million": "Active Cases Per Million",
        "Recovered Per One Million": "Recovered Cases Per Million",
        "Critical Per One Million": "Critical Cases Per Million"}
df3.rename(columns = new_names, inplace=True)


# In[33]:


# 3. For my third change I will simply replace any NaN values with 0
df3.fillna(0, inplace=True) 


# In[34]:


# 4. Next I will remove any duplicates if they are present.
df3.drop_duplicates(inplace=True) 


# In[35]:


# 5. Next I will clean up numeric columns by making them all consistent
numeric_columns = ["Total Cases", "New Cases", "Total Deaths", "Total Recovered", "Active Cases", "Critical Cases",
                       "Cases Per Million", "Deaths Per Million", "Total Tests", "Tests Per Million",
                       "Population", "Active Cases Per Million", "Recovered Cases Per Million",
                       "Critical Cases Per Million"]
for column in numeric_columns: 
    df3[column] = df3[column].astype(int)


# In[36]:


df3


# In[37]:


print(df3.dtypes)


# In[38]:


# Convert DataFrame to CSV file
df3.to_csv('API_data.csv', index=False)

# Read the CSV file
df3 = pd.read_csv('API_data.csv')

# Display the DataFrame
print(df3)


# In[39]:


# Save DataFrame as a new CSV file on the desktop
df3.to_csv('/Users/captainsammy00/Desktop/API_Data.csv', index=False)


# My project is on the 3rd step which is the API portion. It still does not have ethical implications that I need to concern myself with. I was able to extract the data from the API and perform steps to return a clean version of the data. There is more information in this data file because it has different columns when compared to my other website and CSV data. For this API, there were actually multiple columns that I had no use for. After I removed the unnecessary columns I then performed steps that I deemed necessary to clean the data.

# Samuel Castro Project Milestone 5

# In[40]:


import sqlite3
import pandas as pd

# Connect to the SQLite database (create a new one if it doesn't exist)
conn = sqlite3.connect('mydatabase.db')

# Load and create a table for dataset1
CSV_Data = pd.read_csv('Desktop/CSV_Data.csv')
CSV_Data.to_sql('table1', conn, if_exists='replace', index=False)

# Load and create a table for dataset2
Website_Data = pd.read_csv('Desktop/Website_Data.csv')
Website_Data.to_sql('table2', conn, if_exists='replace', index=False)

# Load and create a table for dataset3
API_Data = pd.read_csv('Desktop/API_Data.csv')
API_Data.to_sql('table3', conn, if_exists='replace', index=False)

# Commit the changes and close the database connection
conn.commit()
conn.close()


# In[41]:


conn = sqlite3.connect('mydatabase.db')


# In[42]:


tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
tables


# In[43]:


query = "SELECT * FROM table1"
data = pd.read_sql_query(query, conn)
data.tail()


# In[44]:


query = "SELECT * FROM table2"
data = pd.read_sql_query(query, conn)
data.head()


# In[45]:


query = "SELECT * FROM table3"
data = pd.read_sql_query(query, conn)
data.tail()


# In[46]:


# Reconnect to the SQLite database
conn = sqlite3.connect('mydatabase.db')

# Perform the join operation
query = '''
    SELECT *
    FROM table1
    JOIN table2 ON TRIM((table1."Country/Region")) = TRIM((table2."Country/Region"))
    JOIN table3 ON TRIM((table2."Country/Region")) = TRIM((table3."Country/Region"))
'''

final_dataset = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()


# In[47]:


final_dataset.head()


# In[48]:


# This is my final cleaned dataset
final_dataset


# In[49]:


# Convert DataFrame to CSV file
final_dataset.to_csv('DSC540_Final_Clean_Dataset', index=False)

# Read the CSV file
final_dataset = pd.read_csv('DSC540_Final_Clean_Dataset')

# Display the DataFrame
print(final_dataset)


# In[50]:


final_dataset.to_csv('/Users/captainsammy00/Desktop/DSC540_Final_Clean_Dataset.csv', index=False)


# Visualizations

# In[51]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_Data = pd.read_csv('Desktop/CSV_Data.csv')
Website_Data = pd.read_csv('Desktop/Website_Data.csv')
API_Data = pd.read_csv('Desktop/API_Data.csv')
Final_Dataset = pd.read_csv('Desktop/DSC540_Final_Clean_Dataset.csv')


# In[52]:


CSV_Data.head()


# In[53]:


Website_Data.head()


# In[54]:


API_Data.head()


# In[55]:


Final_Dataset


# In[56]:


# Visualization #1 from Website Data

# Filter the DataFrame to include values less than 500
filtered_df = Website_Data[Website_Data['Deaths Per Million'] < 500]

# Sort the filtered DataFrame by "Deaths Per Million" in descending order
df_sorted = filtered_df.sort_values(by="Deaths Per Million", ascending=False)

# Set the country names as the x-axis and the deaths per million as the y-axis
countries = df_sorted["Country/Region"]
deaths_per_million = df_sorted["Deaths Per Million"]

# Set the width of the bars
bar_width = 0.8

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(countries, deaths_per_million, width=bar_width)
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.xlabel("Country/Region")
plt.ylabel("Deaths Per Million")
plt.title("COVID-19 Deaths Per Million by Country/Region (Values < 500)")
plt.tight_layout()

plt.show()


# In[57]:


# Visualization #2 from Website Data 

# Filter the DataFrame to include values less than 500
filtered_df = Website_Data[Website_Data['Deaths Per Million'] > 4500]

# Sort the filtered DataFrame by "Deaths Per Million" in descending order
df_sorted = filtered_df.sort_values(by="Deaths Per Million", ascending=False)

# Set the country names as labels and the deaths per million as values
countries = df_sorted["Country/Region"]
deaths_per_million = df_sorted["Deaths Per Million"]

# Set explode values to emphasize a particular slice (optional)
explode = [0.1] + [0] * (len(countries) - 1)  # Only explodes the first slice

# Create a pie chart
plt.figure(figsize=(10, 10))  # Increase the figure size
plt.pie(deaths_per_million, labels=countries, explode=explode, autopct='%1.1f%%', textprops={'fontsize': 10})  
plt.title("COVID-19 Deaths Per Million by Country/Region (Values < 500)")
plt.tight_layout()

plt.show()


# In[58]:


# Visualization #3 API Data 

import seaborn as sns

total_cases_million = API_Data['Cases Per Million']
total_deaths_million = API_Data['Deaths Per Million']

# Create a scatter plot

# Define a custom color palette
custom_palette = sns.color_palette('Dark2', n_colors=len(API_Data))

plt.figure(figsize=(16, 10))
sns.scatterplot(x=total_cases_million, y=total_deaths_million, hue=API_Data['Country/Region'], s=100)
plt.xlabel('Total Cases per Million')
plt.ylabel('Total Deaths per Million')
plt.title('Comparison of Total Cases and Total Deaths per Million for Different Regions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# This visualization is not very pleasing to the eye because there are so many countries and this makes the colors very
# indistinguishable. I am only showing this visual because I am going to make a better visual in the next graphs. 


# In[59]:


# Visualization #4 using API data 

import squarify

countries = API_Data['Country/Region']
total_cases = API_Data['Total Cases']

# Sort the data by total cases in descending order
sorted_indices = total_cases.sort_values(ascending=False).index
sorted_countries = countries[sorted_indices]
sorted_total_cases = total_cases[sorted_indices]

# Create a treemap
plt.figure(figsize=(25, 20))
colors = sns.color_palette('tab20')
squarify.plot(sizes=sorted_total_cases, label=sorted_countries, color=colors)
plt.axis('off')
plt.title('Total COVID-19 Cases by Country/Region (Treemap)')
plt.show()

# The purpose of this visualization sole purpose is to show how many cases there are in the USA in 
# comparison to the rest. 


# In[60]:


# Visualization #5 API Data
# Filter the data for the United States
data = API_Data[API_Data['Country/Region'] == 'USA']

# Get the relevant metrics for the United States
Cases_Per_Million = data['Cases Per Million']
Deaths_Per_Million = data['Deaths Per Million']
Active_Cases_Per_Million = data['Active Cases Per Million']
Recovered_Cases_Per_Million = data['Recovered Cases Per Million']
Critical_Cases_Per_Million = data['Critical Cases Per Million']

# Create a bar chart to visualize the metrics for the United States
plt.figure(figsize=(10, 8))
plt.bar('Cases_Per_Million',Cases_Per_Million)
plt.bar('Deaths_Per_Million',Deaths_Per_Million)
plt.bar('Active_Cases_Per_Million',Active_Cases_Per_Million)
plt.bar('Recovered_Cases_Per_Million',Recovered_Cases_Per_Million)
plt.bar('Critical_Cases_Per_Million',Critical_Cases_Per_Million)

plt.xlabel('Metrics')
plt.ylabel('Count')
plt.title('COVID-19 Metrics in the United States')
plt.xticks(rotation=45)
plt.show()

# Here I am showing the specifics of COVID Cases in the USA.


# In[61]:


# Visualization #6 API Data

# Filter the data for the United States
data = API_Data[API_Data['Country/Region'] == 'USA']

# Get the relevant metrics for the United States

Deaths_Per_Million = data['Deaths Per Million']
Active_Cases_Per_Million = data['Active Cases Per Million']

Critical_Cases_Per_Million = data['Critical Cases Per Million']

# Create a bar chart to visualize the metrics for the United States
plt.figure(figsize=(10, 6))

plt.bar('Deaths_Per_Million',Deaths_Per_Million)
plt.bar('Active_Cases_Per_Million',Active_Cases_Per_Million)

plt.bar('Critical_Cases_Per_Million',Critical_Cases_Per_Million)

plt.xlabel('Metrics')
plt.ylabel('Count')
plt.title('COVID-19 Metrics in the United States')
plt.xticks(rotation=45)
plt.show()

# This is a more readable visual of the COVID Cases in the USA.


# In[62]:


# Visual using the combined dataset # 7

plt.figure(figsize=(8, 6))
plt.scatter(Final_Dataset['Active Cases'], Final_Dataset['Recovered Cases'])
plt.title("Active Cases vs. Recovered Cases")
plt.xlabel("Active Cases")
plt.ylabel("Recovered Cases")
plt.show()

# This shows how many recovered cases occur when number of active cases go up. As you can see the less active cases
# the more recovered cases. 


# In[63]:


# Visual using the combined dataset #8

plt.figure(figsize=(8, 6))
plt.scatter(Final_Dataset['Confirmed Cases'], Final_Dataset['Critical Cases'])
plt.title("Confirmed Cases vs. Critical Cases")
plt.xlabel("Confirmed Cases")
plt.ylabel("Critical Cases")
plt.show()

# This shows how small the critical cases are in the grand scheme of things. 


# 250-500 word summary of what you learned and a summary of the ethical implications. 

# This course has been a genuine eye-opener. Before taking this data preparation course, I was doing the bare minimum when it came to data cleaning and manipulation. However, this course has taught me how to clean data in so many different ways, and I have gained a deeper understanding of the importance of data quality and integrity. My project was based on COVID-19 data. This allowed me to find different sources very easily because all of the data was similar to each other. I used countries as my common denominator. 
# 
# One of the key techniques I learned during the course was data cleansing. This involves identifying and correcting or removing errors, inconsistencies, and inaccuracies in datasets. It was emphasized that data cleaning is a crucial step in the data preparation process as it directly impacts the accuracy and reliability of any analysis or model built on that data. I explored various methods to handle missing values, outliers, and duplicate records. I also delved into techniques for standardizing data formats, handling inconsistent data types, and resolving conflicts or contradictions within the dataset. I applied all of these techniques into my first file which was a basic CSV file. 
# 
# The course also touched upon the ethical implications of web data scraping and the use of open APIs. My last two data sources were a specific website from Wikipedia and an API. While web scraping can be a powerful tool for collecting data from websites, it is important to respect the terms and conditions set by website owners and adhere to legal and ethical boundaries. Proper data usage and privacy considerations are critical when scraping data from online sources, and it is essential to be aware of any potential legal issues or violations
# 
# Ethical implications played a significant role throughout the course, especially when it came to data cleansing. While it is essential to ensure the accuracy and reliability of data, we must be mindful of the ethical considerations involved in altering or removing data points. It is crucial to maintain transparency and document any changes made during the data cleansing process. Moreover, any decisions made during data cleaning should be based on sound reasoning and should not introduce bias or alter the integrity of the data.
# 
# In conclusion, this data preparation course has been instrumental in expanding my knowledge and skills in cleaning and preparing data for analysis. I now have a broader understanding of the various techniques and approaches available for data cleansing and integration. Furthermore, I am more conscious of the ethical implications associated with data cleansing, particularly in terms of maintaining data integrity and respecting privacy and legal boundaries. Overall, this course has equipped me with valuable tools and insights that will undoubtedly contribute to more accurate and reliable data analysis and decision-making in the future.

# In[64]:


Final_Dataset


# In[ ]:




