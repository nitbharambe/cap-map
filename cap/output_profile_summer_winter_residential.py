#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
maxi_in_cap=1


# In[73]:


df=pd.read_csv('summer.csv')
print(df.head(721))


# In[74]:


df_diff=df.diff(axis = 0, periods = 1)


# In[75]:


print(df_diff)


# In[76]:


df_diff.index=pd.date_range(start='2019-06-1 00:00:00',end='2019-07-01 23:00:00',freq='h',normalize=True)


# In[77]:


df_diff['perc']=df_diff['kW']/maxi_in_cap


# In[78]:


print(df_diff.head(721))


# In[79]:


df_diff.to_excel(r'C:\Users\15052\Desktop\testing\summer_df.xlsx', sheet_name='table1', index = False)


# In[97]:


summer=pd.read_csv('summer_df_revised.csv')
print(summer.head(720))


# In[102]:


summer_grouped=summer.groupby(['Hourly']).mean()
print(summer_grouped.head(24))


# In[105]:


summer_grouped.to_excel(r'C:\Users\15052\Desktop\testing\summer_too_sort.xlsx', sheet_name='table1', index = True)


# In[126]:


profile_summer_residential=pd.read_csv('summer_sorted.csv')
print(profile_summer_residential.head(24))


# In[44]:


df_winter=pd.read_csv('winter.csv')
print(df_winter.head(745))


# In[108]:


df_winter_diff=df_winter.diff(axis = 0, periods = 1)


# In[109]:


df_winter_diff['perc']=df_winter_diff['kWh.diff-->kW']/maxi_in_cap


# In[110]:


print(df_winter_diff)


# In[111]:


df_winter_diff.index=pd.date_range(start='2019-12-1 00:00:00',end='2020-01-01 23:00:00',freq='h',normalize=True)


# In[112]:


print(df_winter_diff.head(745))


# In[117]:


df_winter_diff.to_excel(r'C:\Users\15052\Desktop\testing\winter_df.xlsx', sheet_name='table1', index = False)


# In[120]:


winter=pd.read_csv('winter_df_revised.csv')
print(winter.head(744))


# In[121]:


winter_grouped=winter.groupby(['Hourly']).mean()
print(winter_grouped.head(24))


# In[124]:


winter_grouped.to_excel(r'C:\Users\15052\Desktop\testing\winter_to_sort.xlsx', sheet_name='table1', index = True)


# In[125]:


profile_winter_residential=pd.read_csv('winter_sorted.csv')
print(profile_winter_residential.head(24))


# In[ ]:




