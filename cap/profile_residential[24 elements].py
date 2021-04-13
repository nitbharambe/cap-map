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


# In[150]:


summer_grouped=summer.groupby(['Hourly']).mean()
print(summer_grouped.head(24))


# In[151]:


summer_grouped.to_excel(r'C:\Users\15052\Desktop\testing\summer_too_sort.xlsx', sheet_name='table1', index = True)


# In[152]:


profile_residential_summer=pd.read_csv('profile_residential_summer_sorted.csv')
print(profile_residential_summer.head(24))


# In[153]:


df_winter=pd.read_csv('profile_residential_winter.csv')
print(df_winter.head(745))


# In[154]:


df_winter_diff=df_winter.diff(axis = 0, periods = 1)


# In[155]:


df_winter_diff['perc']=df_winter_diff['kWh.diff-->kW']/maxi_in_cap


# In[156]:


print(df_winter_diff)


# In[157]:


df_winter_diff.index=pd.date_range(start='2019-12-1 00:00:00',end='2020-01-01 23:00:00',freq='h',normalize=True)


# In[158]:


print(df_winter_diff.head(745))


# In[159]:


df_winter_diff.to_excel(r'C:\Users\15052\Desktop\testing\winter_df.xlsx', sheet_name='table1', index = False)


# In[160]:


winter=pd.read_csv('profile_residential_winter_df_revised.csv')
print(winter.head(744))


# In[161]:


winter_grouped=winter.groupby(['Hourly']).mean()
print(winter_grouped.head(24))


# In[162]:


winter_grouped.to_excel(r'C:\Users\15052\Desktop\testing\winter_to_sort.xlsx', sheet_name='table1', index = True)


# In[163]:


profile_residential_winter=pd.read_csv('profile_residential_winter_sorted.csv')
print(profile_residential_winter.head(24))

