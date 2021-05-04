#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
maxi_in_cap=1


# In[33]:


df=pd.read_csv('profile_residential_summer.csv')
print(df.head(721))


# In[34]:


df_diff=df.diff(axis = 0, periods = 1)


# In[35]:


print(df_diff)


# In[36]:


df_diff.index=pd.date_range(start='2019-06-1 00:00:00',end='2019-07-01 23:00:00',freq='h',normalize=True)


# In[37]:


df_diff['perc']=df_diff['kW']


# In[38]:


print(df_diff.head(721))


# In[39]:


df_diff.to_excel(r'C:\Users\15052\Desktop\testing\summer_df.xlsx', sheet_name='table1', index = False)


# In[40]:


summer=pd.read_csv('profile_residential_summer_df_revised.csv')
print(summer.head(720))


# In[59]:


summer_grouped=summer.groupby(['Hourly']).mean()
print(summer_grouped.head(24))


# In[60]:


summer_grouped.to_excel(r'C:\Users\15052\Desktop\testing\summer_too_sort.xlsx', sheet_name='table1', index = True)


# In[61]:


profile_residential_summer_sorted=pd.read_csv('profile_residential_summer_sorted.csv')
profile_residential_summer_sorted.index=pd.date_range(start='2019-06-1 00:00:00',end='2019-06-1 23:00:00',freq='h',normalize=False)
profile_residential_summer_sorted['perc']=profile_residential_summer_sorted['kW']/maxi_in_cap
print(profile_residential_summer_sorted.head(24))
profile_residential_summer_sorted.to_csv(r'C:\Users\15052\Desktop\testing\profile_residential_summer_24elements.csv')


# In[62]:


df_winter=pd.read_csv('profile_residential_winter.csv')
print(df_winter.head(745))


# In[63]:


df_winter_diff=df_winter.diff(axis = 0, periods = 1)


# In[64]:


df_winter_diff['perc']=df_winter_diff['kWh.diff-->kW']/maxi_in_cap


# In[65]:


print(df_winter_diff)


# In[66]:


df_winter_diff.index=pd.date_range(start='2019-12-1 00:00:00',end='2020-01-01 23:00:00',freq='h',normalize=True)


# In[67]:


print(df_winter_diff.head(745))


# In[68]:


df_winter_diff.to_excel(r'C:\Users\15052\Desktop\testing\winter_df.xlsx', sheet_name='table1', index = False)


# In[69]:


winter=pd.read_csv('profile_residential_winter_df_revised.csv')
print(winter.head(744))


# In[70]:


winter_grouped=winter.groupby(['Hourly']).mean()
print(winter_grouped.head(24))


# In[71]:


winter_grouped.to_excel(r'C:\Users\15052\Desktop\testing\winter_to_sort.xlsx', sheet_name='table1', index = True)


# In[72]:


profile_residential_winter_sorted=pd.read_csv('profile_residential_winter_sorted.csv')
profile_residential_winter_sorted.index=pd.date_range(start='2019-12-1 00:00:00',end='2019-12-1 23:00:00',freq='h',normalize=False)
profile_residential_winter_sorted['perc']=profile_residential_winter_sorted['kW']/maxi_in_cap
print(profile_residential_winter_sorted.head(24))
profile_residential_winter_sorted.to_csv(r'C:\Users\15052\Desktop\testing\profile_residential_winter_24elements.csv')

