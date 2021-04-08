#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
maxi_in_cap=1


# In[68]:


df=pd.read_csv('summer.csv')
print(df.head(721))


# In[69]:


df_diff=df.diff(axis = 0, periods = 1)


# In[70]:


print(df_diff)


# In[71]:


df_diff.index=pd.date_range(start='2019-06-1 00:00:00',end='2019-07-01 23:00:00',freq='h',normalize=True)


# In[72]:


df_diff['perc']=df_diff['kWh.diff-->kW']/maxi_in_cap


# In[73]:


print(df_diff.head(721))


# In[74]:


df_winter=pd.read_csv('winter.csv')
print(df_winter.head(745))


# In[75]:


df_winter_diff=df_winter.diff(axis = 0, periods = 1)


# In[76]:


df_winter_diff['perc']=df_winter_diff['kWh.diff-->kW']/maxi_in_cap


# In[77]:


print(df_winter_diff)


# In[60]:


df_winter_diff.index=pd.date_range(start='2019-12-1 00:00:00',end='2020-01-01 23:00:00',freq='h',normalize=True)


# In[61]:


print(df_winter_diff.head(745))


# In[ ]:




