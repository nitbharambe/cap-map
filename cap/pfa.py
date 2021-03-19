#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


#Install libraries if using google colab
#pip install pandapower
#pip install simbench


# In[1]:


#Import the pandapower and the networks module:
import pandapower as pp
import pandapower.networks as nw
import simbench as sb
import pandas as pd
import numpy as np
import os,sys
'''
#Plotting
from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
from pandapower.plotting.plotly import simple_plotly, pf_res_plotly, vlevel_plotly
import matplotlib.pyplot as plt
import pandapower.plotting as plot
import pandapower.plotting.plotly as pplotly
try:
    import seaborn
    colors = seaborn.color_palette()
except:
    colors = ["b", "g", "r", "c", "y"]
%matplotlib inline
'''
#Timeseries
import tempfile
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries


# ## Define Functions

# In[2]:


def init_net():
    '''
    Initialization of load and generation p_mw and q_mvar is needed because the run_timeseries replaces them after every iteration
    
    OUTPUT - 
        net- Pandapower network with initial values
    '''
    net.load[['p_mw','q_mvar']]=initload
    net.sgen[['p_mw','q_mvar']]=initsgen
    return net

def define_log():
    '''
    Creates output writer object required for timeseries simulation
    The timeseries module only calculates the values of variables mentioned here for each simulation. 
    The temporary data gets stored in the output_dir directory
    
    OUTPUT
        ow - Output writer object
    '''
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".json")    
    ow.log_variable('res_bus','vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'loading_percent')
    return ow

def add_loadgen(net_t, loadorgen, conn_at_bus, size_p,size_q,prof):
    '''
    Adds a load or generation to the net.load or net.sgen table. 
    Adds profile name to the profiles variable of the newly addded capacity.

    INPUT
        net_t (PP net) - Pandapower net
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        conn_at_bus (int) - Bus at which additional capacity is connected
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid
        
    OUTPUT
        net_t (PP net) - Updated Pandapower net
        
    '''
    if loadorgen=="load":
        pp.create_load(net_t, conn_at_bus, p_mw=size_p, q_mvar=size_q)
        net_t.load.tail(1).profile=prof
    elif loadorgen=="sgen":
        pp.create_sgen(net_t, conn_at_bus, p_mw=size_p, q_mvar=size_q)
        net_t.sgen.tail(1).profile=prof
    else:
        return 0
    return net_t

def drop_loadgen(net_t, loadorgen):
    '''
    Removes the last added capacity from the input grid added by add_loadgen
    
    INPUT
        net_t (PP net) - Pandapower net
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        
    OUTPUT
        net_t (PP net) - Updated Pandapower net
    '''
    if loadorgen=="load":
        net_t.load=net_t.load.head(-1)            
    elif loadorgen=="sgen":
        net_t.sgen=net_t.sgen.head(-1)
    return net_t

def violations(net):
    '''
    Checks for any violations created in the grid by additional capacity.
    Loads the files created by timeseries simulation. Compares simulation values against the limits mentioned in the input grid.
    
    TODO: Create separate violations function for individual input. (Already done, might have to include it here)
    
    INPUT
        net (PP net) - Pandapower net
        
    OUTPUT
        check (bool) - 'True' for no violations. 'False' for violations present
        
    '''
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.json")
    vm_pu = pd.read_json(vm_pu_file)

    line_load_file = os.path.join(output_dir, "res_line", "loading_percent.json")
    line_load = pd.read_json(line_load_file)

    trafo_load_file = os.path.join(output_dir, "res_trafo", "loading_percent.json")
    trafo_load = pd.read_json(trafo_load_file)

    check = any(np.where(vm_pu.max() > net.bus['max_vm_pu'],True, False))
    check = check or any(np.where(vm_pu.min() < net.bus['min_vm_pu'],True, False))
    check = check or any(np.where(line_load.max() > net.line['max_loading_percent'],True, False))
    check = check or any(np.where(trafo_load.max() > net.trafo['max_loading_percent'],True, False))
    return not check

def feas_chk(net,ow,conn_at_bus,loadorgen, size_p, size_q, prof):
    '''
    Initializes the PPnet, 
    Adds additional capacity, 
    applies load/generation profiles on all the grid elements,
    runs timeseries for the specific case and save the results in the temporary output directory, 
    clears the application of profiles from the input grid (Deletes the Constcontrol objects), 
    drops the added capacity.

    BUG: More like pending to do. Doesnt work for loads. Need to check process of how profiles from simbench are actually
    getting applied to Constcontrol know the fix. Also will lead to finding how profiles from input will be applied on 
    the input grid.
    TODO: suppress/workaround printing of individual progress bars
    
    INPUT
        net (PP net) - Pandapower net
        ow (Object) - Output writer object
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        conn_at_bus (int) - Bus at which additional capacity is connected
        size_p (int) - Size of active power of additional capacity
        size_q (int) - Size of reactive power of additional capacity
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid
        
    OUTPUT
        feas_result (bool) - 'True' for feasible, 'False' for not feasible
        
    '''
    net=init_net()
    net=add_loadgen(net, loadorgen, conn_at_bus, size_p,size_q, prof)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    sb.apply_const_controllers(net, profiles)    #create timeseries data from profiles and run powerflow
    run_timeseries(net,time_steps,verbose=True)               #Run powerflow only over time_steps
    net.controller=net.controller.iloc[0:0]
    net=drop_loadgen(net, loadorgen)
    feas_result=violations(net)
    return feas_result

def max_cap(net,ow,conn_at_bus,loadorgen,ul_p,ll_p,prof):
    '''
    Seach algorithm using feas_chk function over the range of ll_p and ul_p capacities
    BUG: Doesnt work for capacity below 20MW, must require a minor fix
    
    INPUT
        net (PP net) - Pandapower net
        ow (Object) - Output writer object
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        conn_at_bus (int) - Bus at which additional capacity is connected
        ll_p (int) - Size of maximum power limit of additional capacity that can be added
        ul_p (int) - Size of minimum additional capacity that can be added (Set as 0)
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid
        
    OUTPUT
         (int) - Maximum capacitiy of load/generation that can be added at given bus
        
    '''
    no_iter=0
    [ul_chk,mid_chk,ll_chk]=False,False,False
    while not( ((ul_p-ll_p)<s_tol) | (ul_chk & mid_chk) | (no_iter>7) ):
        no_iter=no_iter+1
        mid_p=(ul_p+ll_p)/2
        ul_chk=feas_chk(net,ow,conn_at_bus,loadorgen, size_p=ul_p, size_q=inp_q,prof=prof)
        mid_chk=feas_chk(net,ow,conn_at_bus,loadorgen, size_p=mid_p, size_q=inp_q,prof=prof)
        if mid_chk==True:
            ll_p=mid_p
        elif mid_chk==False:
            ul_p=mid_p
        elif ul_chk:
            return ul_p
            break
    return ll_p

def all_cap_map(net,ow,loadorgen,ul_p,ll_p,prof):
    '''
    Iteratre the max_cap function over all busses in the grid. 
    
    TODO: Add progess bar
    
    INPUT
        net (PP net) - Pandapower net
        ow (Object) - Output writer object
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        ll_p (int) - Size of maximum power limit of additional capacity that can be added
        ul_p (int) - Size of minimum additional capacity that can be added (Set as 0)
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid
        
    OUTPUT
         allcap (dataframe) - Maximum capacitiy of load/generation that can be added at all buses
    
    '''    
    allcap=net.bus[['name','vn_kv']]
    allcap['max_add_cap']=np.nan
    for conn_at_bus in range(len(net.bus)):
        allcap['max_add_cap'][conn_at_bus]=max_cap(net,ow=ow,conn_at_bus=conn_at_bus,loadorgen=loadorgen,ul_p=ul_p,ll_p=ll_p,prof=prof)
    return allcap


# ## Set Parameters

# ### Input Grid

# In[3]:


'''
Input grid for developing code is sb_code1
sb_code2 is the larger grid to test the code
Initial values of the loads and generation are stored
Maximum voltage limits are relaxed for testing sample code since the limit gets violated for the test grid without adding any capacity
'''
sb_code1 = "1-MV-rural--1-sw"  # rural MV grid of scenario 0 with full switchs
sb_code2 = "1-HVMV-urban-all-0-sw"  # urban hv grid with one connected mv grid which has the subnet 2.202
net = sb.get_simbench_net(sb_code1)
net.bus.max_vm_pu=net.bus.max_vm_pu*1.05
initload=net.load[['p_mw','q_mvar']]
initsgen=net.sgen[['p_mw','q_mvar']]


# ### Other parameters

# In[4]:


'''
time_steps:Set time steps in range for the timeseries module to compute over. 
    This parameter must be of same length as the length of profiles.
ll_p and ul_p : limits for maximum and minimum capacity that can be added to any bus
inp_q: input reactive power for added capacity. Assumed constant
tol: Search algorithm tolerance (in MW)
output_dir : Set directory for storing the logged varaiables. 
    Commented output_dir line is for setting directory in the temporary files of the computer.
ow: Create the output writer object ow
'''
time_steps=range(96)
ll_p=20
ul_p=90
inp_q=0.1
s_tol=1

#output_dir = os.path.join(tempfile.gettempdir(), "simp_cap_v3")
output_dir = os.path.join('C:\\Users\\nitbh\\OneDrive\\Documents\\IIPNB', "simp_cap_v3")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
ow=define_log()   #For logging variables


# ## Set Input

# In[5]:


'''
Sample input. Use as needed
prof and loadorgen are needed for getting the map
Others are needed in case of individual capacity checks
'''
#size_pmw=10
#size_qmw=0.05
loadorgen='sgen'
#prof='L0-A'
prof='WP4'
#conn_at_bus=2


# ## Get Map

# All_cap_map takes lot of time to calculate capacities of an entire map. Try using inner functions one at a time

# In[6]:




# In[7]:
'''

#max_cap(net,ow=ow,conn_at_bus=2, loadorgen='sgen',ul_p=ul_p, ll_p=ll_p, prof='WP4')

from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput

graphviz = GraphvizOutput()
graphviz.output_file = 'basic.png'
with PyCallGraph(output=GraphvizOutput()):
    max_cap(net, ow=ow, conn_at_bus=2, loadorgen='sgen', ul_p=ul_p, ll_p=ll_p, prof='WP4')
    #code_to_profile()
'''