#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
maxi_in_cap=1


# In[2]:


df=pd.read_csv('profile_residential_summer.csv')
print(df.head(721))


# In[3]:


df_diff=df.diff(axis = 0, periods = 1)


# In[4]:


print(df_diff)


# In[5]:


df_diff.index=pd.date_range(start='2019-06-1 00:00:00',end='2019-07-01 23:00:00',freq='h',normalize=True)


# In[6]:


df_diff['perc']=df_diff['kW']/maxi_in_cap


# In[7]:


print(df_diff.head(721))


# In[8]:


df_diff.to_excel(r'C:\Users\15052\Desktop\testing\summer_df.xlsx', sheet_name='table1', index = False)


# In[9]:


summer=pd.read_csv('profile_residential_summer_df_revised.csv')
print(summer.head(720))


# In[19]:


summer_grouped=summer.groupby(['Hourly']).mean()
print(summer_grouped.head(24))


# In[20]:


summer_grouped.to_excel(r'C:\Users\15052\Desktop\testing\summer_too_sort.xlsx', sheet_name='table1', index = True)


# In[23]:


profile_residential_summer_sorted=pd.read_csv('profile_residential_summer_sorted.csv')
profile_residential_summer_sorted.index=pd.date_range(start='2019-06-1 00:00:00',end='2019-06-1 23:00:00',freq='h',normalize=False)
print(profile_residential_summer_sorted.head(24))


# In[24]:


df_winter=pd.read_csv('profile_residential_winter.csv')
print(df_winter.head(745))


# In[25]:


df_winter_diff=df_winter.diff(axis = 0, periods = 1)


# In[26]:


df_winter_diff['perc']=df_winter_diff['kWh.diff-->kW']/maxi_in_cap


# In[27]:


print(df_winter_diff)


# In[28]:


df_winter_diff.index=pd.date_range(start='2019-12-1 00:00:00',end='2020-01-01 23:00:00',freq='h',normalize=True)


# In[29]:


print(df_winter_diff.head(745))


# In[30]:


df_winter_diff.to_excel(r'C:\Users\15052\Desktop\testing\winter_df.xlsx', sheet_name='table1', index = False)


# In[160]:


winter=pd.read_csv('profile_residential_winter_df_revised.csv')
print(winter.head(744))


# In[161]:


winter_grouped=winter.groupby(['Hourly']).mean()
print(winter_grouped.head(24))


# In[162]:


winter_grouped.to_excel(r'C:\Users\15052\Desktop\testing\winter_to_sort.xlsx', sheet_name='table1', index = True)


# In[31]:


profile_residential_winter_sorted=pd.read_csv('profile_residential_winter_sorted.csv')
profile_residential_winter_sorted.index=pd.date_range(start='2019-12-1 00:00:00',end='2019-12-1 23:00:00',freq='h',normalize=False)
print(profile_residential_winter_sorted.head(24))

