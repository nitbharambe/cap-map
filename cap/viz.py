#Install libraries if using google colab
%pip install pandapower
%pip install simbench

#Import the pandapower and the networks module:
import pandapower as pp
import pandapower.networks as nw
import simbench as sb
import pandas as pd
import numpy as np
import os,sys

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

#Timeseries
import tempfile
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries

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

    BUG: More like pending TODO. Doesnt work for loads. Need to check process of how profiles from simbench are actually 
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
    Iterate the max_cap function over all busses in the grid. 
    
    TODO: Add overall progess bar
    TODO: Speed up simulation, if needed: find execution time of each step

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

output_dir = os.path.join(tempfile.gettempdir(), "simp_cap_v3")
#output_dir = os.path.join('C:\\Users\\nitbh\\OneDrive\\Documents\\IIPNB', "simp_cap_v3")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
ow=define_log()   #For logging variables

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

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
 
 
import pandas as pd
 
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.mapbox_plot import *
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace, draw_traces, version_check
from pandapower.run import runpp
 
try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)
 
 
def pf_res_plotly1(net, cmap="Jet", use_line_geodata=None, on_map=False, projection=None,
                  map_style='basic', figsize=1, aspectratio='auto', line_width=2, bus_size=10,
                  climits_volt=(0.9, 1.1), climits_load=(0, 100), cpos_volt=1.0, cpos_load=1.1,
                  filename="temp-plot.html"):
    """
    Plots a pandapower network in plotly
    using colormap for coloring lines according to line loading and buses according to voltage in p.u.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial
 
    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example
 
    OPTIONAL:
        **respect_switches** (bool, False) - Respect switches when artificial geodata is created
 
        *cmap** (str, True) - name of the colormap
 
        *colors_dict** (dict, None) - by default 6 basic colors from default collor palette is used.
        Otherwise, user can define a dictionary in the form: voltage_kv : color
 
        **on_map** (bool, False) - enables using mapbox plot in plotly
        If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.
 
        **projection** (String, None) - defines a projection from which network geo-data will be transformed to
        lat-long. For each projection a string can be found at http://spatialreference.org/ref/epsg/
 
        **map_style** (str, 'basic') - enables using mapbox plot in plotly
 
            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'
 
        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size
 
        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the network geodata
        any custom aspectration can be given as a tuple, e.g. (1.2, 1)
 
        **line_width** (float, 1.0) - width of lines
 
        **bus_size** (float, 10.0) -  size of buses to plot.
 
        **climits_volt** (tuple, (0.9, 1.0)) - limits of the colorbar for voltage
 
        **climits_load** (tuple, (0, 100)) - limits of the colorbar for line_loading
 
        **cpos_volt** (float, 1.0) - position of the bus voltage colorbar
 
        **cpos_load** (float, 1.1) - position of the loading percent colorbar
 
        **filename** (str, "temp-plot.html") - filename / path to plot to. Should end on *.html
 
    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object
 
    """
    version_check()
    if 'res_bus' not in net or net.get('res_bus').shape[0] == 0:
        logger.warning('There are no Power Flow results. A Newton-Raphson power flow will be executed.')
        runpp(net)
 
    # create geocoord if none are available
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=True)
        if on_map:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False
 
    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)
 
    # ----- Buses ------
    # initializating bus trace
    # hoverinfo which contains name and pf results
    precision = 3
    hoverinfo = (
            net.bus.name.astype(str) + '<br />' +
            'V_m = ' + net.res_bus.vm_pu.round(precision).astype(str) + 'pu' + '<br />' +
            'V_m = ' + (net.res_bus.vm_pu * net.bus.vn_kv.round(2)).round(precision).astype(str) + ' kV' + '<br />' +
            'Max_cap =' + net.load.max_load.astype(str) + 'MW' + '<br />' +
            'V_a = ' + net.res_bus.va_degree.round(precision).astype(str) + ' deg').tolist()
    hoverinfo = pd.Series(index=net.bus.index, data=hoverinfo)
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, infofunc=hoverinfo, cmap=cmap,
                                 cbar_title='Bus Voltage [pu]', cmin=climits_volt[0], cmax=climits_volt[1],
                                 cpos=cpos_volt)
 
    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    # if bus geodata is available, but no line geodata
    cmap_lines = 'jet' if cmap == 'Jet' else cmap
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False
    # hoverinfo which contains name and pf results
    hoverinfo = (
            net.line.name.astype(str) + '<br />' +
            'I = ' + net.res_line.loading_percent.round(precision).astype(str) + ' %' + '<br />' +
            'I_from = ' + net.res_line.i_from_ka.round(precision).astype(str) + ' kA' + '<br />' +
            'I_to = ' + net.res_line.i_to_ka.round(precision).astype(str) + ' kA' + '<br />').tolist()
    hoverinfo = pd.Series(index=net.line.index, data=hoverinfo)
    line_traces = create_line_trace(net, use_line_geodata=use_line_geodata, respect_switches=True,
                                    width=line_width,
                                    infofunc=hoverinfo,
                                    cmap=cmap_lines,
                                    cmap_vals=net.res_line['loading_percent'].values,
                                    cmin=climits_load[0],
                                    cmax=climits_load[1],
                                    cbar_title='Line Loading [%]',
                                    cpos=cpos_load)
 
    # ----- Trafos ------
    # hoverinfo which contains name and pf results
    hoverinfo = (
            net.trafo.name.astype(str) + '<br />' +
            'I = ' + net.res_trafo.loading_percent.round(precision).astype(str) + ' %' + '<br />' +
            'I_hv = ' + net.res_trafo.i_hv_ka.round(precision).astype(str) + ' kA' + '<br />' +
            'I_lv = ' + net.res_trafo.i_lv_ka.round(precision).astype(str) + ' kA' + '<br />').tolist()
    hoverinfo = pd.Series(index=net.trafo.index, data=hoverinfo)
    trafo_traces = create_trafo_trace(net, width=line_width * 1.5, infofunc=hoverinfo,
                                      cmap=cmap_lines, cmin=0, cmax=100)
 
    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color='grey', size=bus_size * 2, trace_name='external_grid',
                                      patch_type=marker_type)
 
    return draw_traces(line_traces + trafo_traces + ext_grid_trace + bus_trace,
                       showlegend=False, aspectratio=aspectratio, on_map=on_map,
                       map_style=map_style, figsize=figsize, filename=filename)

net.sgen['max_sgen']=np.random.randint(0,100,net.sgen.shape[0])
net.load['max_load']=np.random.randint(0,100,net.load.shape[0])

#from pandapower.plotting.plotly import pf_res_plotly
#from pandapower.networks import mv_oberrhein
sb_code1 = "1-MV-rural--1-sw"  # rural MV grid of scenario 0 with full switchs
#sb_code2 = "1-HVMV-urban-all-0-sw"  # urban hv grfid with one connected mv grid which has the subnet 2.202
#net = sb.get_simbench_net(sb_code1)
pf_res_plotly1(net)
