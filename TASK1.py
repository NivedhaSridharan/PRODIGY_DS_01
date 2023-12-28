#!/usr/bin/env python
# coding: utf-8

# ![Screenshot%202023-07-11%20200705.png](attachment:Screenshot%202023-07-11%20200705.png)

# ![Screenshot%202023-07-20%20203649.png](attachment:Screenshot%202023-07-20%20203649.png)

# # | About Dataset
# * Series Name
# * Series Code:Contain records of total population, male population and female population
# * Country Name
# * Country Code
# * 2022-2001:22 columns having year wise records

#  # --------------------------------------------- Table of Content  ---------------------------------------
# | No | Content |
# |-----|--------|
# | 1 | Introduction |
# | 2 | Data Overview |
# | 3 | Data Cleaning |
# | 4 | Data Visualization |

# #  1 | Introduction
# ## I | Import libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## II | Import data

# In[2]:


data=pd.read_csv('worldpopulationdata.csv')


# In[3]:


data.head()


# # 2 | Data Overview

# In[4]:


print(f'''The shape of data: {data.shape}''')


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# # 3 | Data Cleaning
# ## 1- Drop unuseful columns
# 

# In[8]:


data = data.drop("Country Code", axis=1)
data = data.drop("Series Name", axis=1)


# ## 2- Extracted top ten countries of total population

# In[9]:


# Filter data for total population
total_population_data = data[data["Series Code"] == "SP.POP.TOTL"]

# Sort data based on the total population for 2022
total_population_sorted = total_population_data.sort_values(by="2022", ascending=False)

# Get the top ten countries with the highest total population for 2022
total_top_ten_countries = total_population_sorted.head(10)
print("Top ten countries of total population")
print(total_top_ten_countries )


# # 4 | Data Visualization

# In[10]:


def add_value_labels(ax, spacing=10):

    # For each bar: Place a label    
    for rect in ax.patches:
        
        # Get X and Y placement of label from rect.
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()-3

        # Determine vertical alignment for positive and negative values
        va = 'bottom' if y >= 0 else 'top'

        # Format the label to one decimal place
        label = "{}".format(y)

        # Determine the vertical shift of the label
        # based on the sign of the y value and the spacing parameter
        y_shift = spacing * (1 if y >= 0 else -1)

        # Create the annotation
        ax.annotate(label, (x, y), xytext=(0, y_shift),textcoords="offset points", ha='center', va=va)


# ## 1- Bar Plot
# ### Top ten countries of total population in year 2022 and 2016

# In[11]:


# Create the bar plot
plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2022", y="Country Name", data=total_top_ten_countries, palette="coolwarm")
plt.title("Top Ten Countries of Total Population (2022)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2016", y="Country Name", data=total_top_ten_countries, palette="coolwarm")
plt.title("Top Ten Countries with Total Population (2016)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries of total population in year 2010 and 2001

# In[12]:


# Create the bar plot
plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2010", y="Country Name", data=total_top_ten_countries, palette="coolwarm")
plt.title("Top Ten Countries of Total Population (2010)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2001", y="Country Name", data=total_top_ten_countries, palette="coolwarm")
plt.title("Top Ten Countries with Total Population (2001)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ## 3- Extracted bottom ten countries of total population

# In[13]:


# Sort data based on the total population for 2022
total_population_sorted1 = total_population_data.sort_values(by="2022", ascending=True)

# Get the bottom ten countries of total population for 2022
total_bottom_ten_countries = total_population_sorted1.head(10)
print("Bottom ten countries of total population")
print(total_bottom_ten_countries )


# ### Bottom ten countries of total population in year 2022 and 2016

# In[14]:


# Create the bar plot
plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2022", y="Country Name", data=total_bottom_ten_countries, palette="coolwarm")
plt.title("Bottom Ten Countries of Total Population (2022)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2016", y="Country Name", data=total_bottom_ten_countries, palette="coolwarm")
plt.title("Bottom Ten Countries with Total Population (2016)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Bottom ten countries of total population in year 2010 and 2001

# In[15]:


# Create the bar plot
plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2010", y="Country Name", data=total_bottom_ten_countries, palette="coolwarm")
plt.title("Bottom Ten Countries of Total Population (2010)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2001", y="Country Name", data=total_bottom_ten_countries, palette="coolwarm")
plt.title("Bottom Ten Countries with Total Population (2001)")
plt.xlabel("Total Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ## 4- Extracted top ten countries with highest male population

# In[16]:


# Filter data for male population
male_population_data = data[data["Series Code"] == "SP.POP.TOTL.MA.IN"]

# Sort data based on the male population for 2022
male_population_sorted = male_population_data.sort_values(by="2022", ascending=False)

# Get the top ten countries with the highest male population for 2022
male_top_ten_countries = male_population_sorted.head(10)
print("Top ten countries of male population")
print(male_top_ten_countries )


# ## 5- Extracted top ten countries with highest female population

# In[17]:


# Filter data for female population
female_population_data = data[data["Series Code"] == "SP.POP.TOTL.FE.IN"]

# Sort data based on the female population for 2022
female_population_sorted = female_population_data.sort_values(by="2022", ascending=False)

# Get the top ten countries with the highest female population for 2022
female_top_ten_countries = female_population_sorted.head(10)

print("Top ten countries of female population")
print(female_top_ten_countries)


# ### Top ten countries with highest male and female population in 2022

# In[18]:


# Create the bar plot
plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2022", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2022)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2022", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Femaale Population (2022)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with highest male and female population in 2021

# In[19]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2021", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2021)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2021", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Female Population (2021)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with highest male and female population in 2020

# In[20]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2020", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2020)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2020", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Female Population (2020)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with highest male and female population in 2019

# In[21]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2019", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2019)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2019", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Female Population (2019)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with highest male and female population in 2016

# In[22]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2016", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2016)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2016", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Female Population (2016)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with highest male and female population in 2010

# In[23]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2010", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2010)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2010", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Female Population (2010)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with highest male and female population in 2006

# In[24]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2006", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2006)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2006", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Female Population (2006)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with highest male and female population in 2001

# In[25]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2001", y="Country Name", data=male_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Male Population (2001)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2001", y="Country Name", data=female_top_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Highest Female Population (2001)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ## 6. Extracted top ten countries with lowest male population

# In[26]:


# Sort data based on the male population for 2022 in ascending order
male_population_sorted = male_population_data.sort_values(by="2022", ascending=True)

# Get the top ten countries with the lowest male population for 2022
male_lowest_ten_countries = male_population_sorted.head(10)

print("Top ten countries with lowest male population")
print(male_lowest_ten_countries )


# ## 7. Extracted top ten countries with lowest female population

# In[27]:


# Sort data based on the female population for 2022 in ascending order
female_population_sorted = female_population_data.sort_values(by="2022", ascending=True)

# Get the top ten countries with the lowest female population for 2022
female_lowest_ten_countries = female_population_sorted.head(10)

print("Top ten countries with lowest female population")
print(female_lowest_ten_countries )


# ### Top ten countries with lowest male and female population in 2022

# In[28]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2022", y="Country Name", data=male_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Male Population (2022)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2022", y="Country Name", data=female_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Female Population (2022)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with lowest male and female population in 2018

# In[29]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2018", y="Country Name", data=male_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Male Population (2018)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2018", y="Country Name", data=female_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Female Population (2018)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with lowest male and female population in 2012

# In[30]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2012", y="Country Name", data=male_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Male Population (2012)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2012", y="Country Name", data=female_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Female Population (2012)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with lowest male and female population in 2008

# In[31]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2008", y="Country Name", data=male_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Male Population (2008)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2008", y="Country Name", data=female_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Female Population (2008)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ### Top ten countries with lowest male and female population in 2001

# In[32]:


plt.figure(figsize=(20, 16))
plt.rcParams['axes.facecolor'] = 'black'

plt.subplot(2,2,1)
plot=sns.barplot(x="2001", y="Country Name", data=male_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Male Population (2001)")
plt.xlabel("Male Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.subplot(2,2,2)
plot=sns.barplot(x="2001", y="Country Name", data=female_lowest_ten_countries, palette="viridis")
plt.title("Top Ten Countries with Lowest Female Population (2001)")
plt.xlabel("Female Population")
plt.ylabel("Country")
add_value_labels(plot)

plt.tight_layout()


# ## Stacked Bar Plot 
# ### Top 10 Countries with Male and Female Populations (2022)

# In[33]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2022_male"] + merged_data["2022_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=False)

# Select the top 10 countries with the highest total population
top_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2022_female", data=top_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2022_male", data=top_10_countries, bottom=top_10_countries["2022_female"], color="blue", label="Male Population")
plt.title("Top 10 Countries with Male and Female Populations (2022)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ### Top 10 Countries with Male and Female Populations (2016)

# In[34]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2016_male"] + merged_data["2016_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=False)

# Select the top 10 countries with the highest total population
top_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2016_female", data=top_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2016_male", data=top_10_countries, bottom=top_10_countries["2016_female"], color="blue", label="Male Population")
plt.title("Top 10 Countries with Male and Female Populations (2016)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ### Top 10 Countries with Male and Female Populations (2010)

# In[35]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2010_male"] + merged_data["2010_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=False)

# Select the top 10 countries with the highest total population
top_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2010_female", data=top_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2010_male", data=top_10_countries, bottom=top_10_countries["2010_female"], color="blue", label="Male Population")
plt.title("Top 10 Countries with Male and Female Populations (2010)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ### Top 10 Countries with Male and Female Populations (2001)

# In[36]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2001_male"] + merged_data["2001_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=False)

# Select the top 10 countries with the highest total population
top_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2001_female", data=top_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2001_male", data=top_10_countries, bottom=top_10_countries["2001_female"], color="blue", label="Male Population")
plt.title("Top 10 Countries with Male and Female Populations (2001)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ### Bottom 10 Countries with Male and Female Populations (2022)

# In[37]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2022_male"] + merged_data["2022_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=True)

# Select the top 10 countries with the highest total population
bottom_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2022_female", data=bottom_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2022_male", data=bottom_10_countries, bottom=bottom_10_countries["2022_female"], color="blue", label="Male Population")
plt.title("Bottom 10 Countries with Male and Female Populations (2022)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ### Bottom 10 Countries with Male and Female Populations (2016)

# In[38]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2016_male"] + merged_data["2016_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=True)

# Select the top 10 countries with the highest total population
bottom_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2016_female", data=bottom_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2016_male", data=bottom_10_countries, bottom=bottom_10_countries["2016_female"], color="blue", label="Male Population")
plt.title("Bottom 10 Countries with Male and Female Populations (2016)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ### Bottom 10 Countries with Male and Female Populations (2010)

# In[39]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2010_male"] + merged_data["2010_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=True)

# Select the top 10 countries with the highest total population
bottom_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2010_female", data=bottom_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2010_male", data=bottom_10_countries, bottom=bottom_10_countries["2010_female"], color="blue", label="Male Population")
plt.title("Bottom 10 Countries with Male and Female Populations (2010)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ### Bottom 10 Countries with Male and Female Populations (2001)

# In[40]:


# Merge male and female population data on 'Country Name'
merged_data = pd.merge(male_population_data, female_population_data, on="Country Name", suffixes=("_male", "_female"))

# Calculate the total population for each country (male + female)
merged_data["Total Population"] = merged_data["2001_male"] + merged_data["2001_female"]

# Sort data based on total population in descending order
sorted_data = merged_data.sort_values(by="Total Population", ascending=True)

# Select the top 10 countries with the highest total population
bottom_10_countries = sorted_data.head(10)


# Set seaborn style
sns.set(style="whitegrid")

# Create the stacked bar plot
plt.figure(figsize=(12, 6))

sns.barplot(x="Country Name", y="2001_female", data=bottom_10_countries, color="pink", label="Female Population")
sns.barplot(x="Country Name", y="2001_male", data=bottom_10_countries, bottom=bottom_10_countries["2001_female"], color="blue", label="Male Population")
plt.title("Bottom 10 Countries with Male and Female Populations (2001)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:




