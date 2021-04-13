#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
maxi_in_cap=1


# In[128]:


df=pd.read_csv('profile_residential_summer.csv')
print(df.head(721))


# In[129]:


df_diff=df.diff(axis = 0, periods = 1)


# In[130]:


print(df_diff)


# In[131]:


df_diff.index=pd.date_range(start='2019-06-1 00:00:00',end='2019-07-01 23:00:00',freq='h',normalize=True)


# In[132]:


df_diff['perc']=df_diff['kW']/maxi_in_cap


# In[133]:


print(df_diff.head(721))


# In[134]:


df_diff.to_excel(r'C:\Users\15052\Desktop\testing\summer_df.xlsx', sheet_name='table1', index = False)


# In[135]:


summer=pd.read_csv('profile_residential_summer_df_revised.csv')
print(summer.head(720))


# In[136]:


summer_grouped=summer.groupby(['Hourly']).mean()
print(summer_grouped.head(24))


# In[137]:


summer_grouped.to_excel(r'C:\Users\15052\Desktop\testing\summer_too_sort.xlsx', sheet_name='table1', index = True)


# In[138]:


profile_summer_residential=pd.read_csv('profile_residential_summer_sorted.csv')
print(profile_summer_residential.head(24))


# In[139]:


df_winter=pd.read_csv('profile_residential_winter.csv')
print(df_winter.head(745))


# In[140]:


df_winter_diff=df_winter.diff(axis = 0, periods = 1)


# In[141]:


df_winter_diff['perc']=df_winter_diff['kWh.diff-->kW']/maxi_in_cap


# In[142]:


print(df_winter_diff)


# In[143]:


df_winter_diff.index=pd.date_range(start='2019-12-1 00:00:00',end='2020-01-01 23:00:00',freq='h',normalize=True)


# In[144]:


print(df_winter_diff.head(745))


# In[145]:


df_winter_diff.to_excel(r'C:\Users\15052\Desktop\testing\winter_df.xlsx', sheet_name='table1', index = False)


# In[146]:


winter=pd.read_csv('profile_residential_winter_df_revised.csv')
print(winter.head(744))


# In[147]:


winter_grouped=winter.groupby(['Hourly']).mean()
print(winter_grouped.head(24))


# In[148]:


winter_grouped.to_excel(r'C:\Users\15052\Desktop\testing\winter_to_sort.xlsx', sheet_name='table1', index = True)


# In[149]:


profile_winter_residential=pd.read_csv('profile_residential_winter_sorted.csv')
print(profile_winter_residential.head(24))

