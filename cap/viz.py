#!/usr/bin/env python
# coding: utf-8

# My note (Joe):
# 1. fixed the graph function for general users (run through all the data)

# ### 1. preprocessing the data

# In[5]:


# power flow package & python package ( edited by Joe )
from cap import pfa
import copy
import pandas as pd

# Dash import
import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# panda package
from pandapower.plotting.plotly import pf_res_plotly
import numpy as np

# for copy panda object
from copy import deepcopy

# for editing the draw_trace() function
from pandapower.plotting.plotly import traces
from pandapower.plotting.plotly.traces import _in_ipynb

import math

from packaging import version
from collections.abc import Iterable

from pandapower.plotting.plotly.get_colors import get_plotly_color, get_plotly_cmap
from pandapower.plotting.plotly.mapbox_plot import _on_map_test, _get_mapbox_token,     MapboxTokenMissing

try:
    from plotly import __version__ as plotly_version
    from plotly.graph_objs.scatter.marker import ColorBar
    from plotly.graph_objs import Figure, Layout
    from plotly.graph_objs.layout import XAxis, YAxis
    from plotly.graph_objs.scatter import Line, Marker
    from plotly.graph_objs.scattermapbox import Line as scmLine
    from plotly.graph_objs.scattermapbox import Marker as scmMarker
except ImportError:
    logger.info("Failed to import plotly - interactive plotting will not be available")

# changing the function part (edited by Daniel)
# import pandas as pd 
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.mapbox_plot import *
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace,     create_trafo_trace, draw_traces, version_check
from pandapower.run import runpp
 
try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


# In[2]:


# Edited function res_plotly() and draw_traces()

def draw_traces_nograph(traces, on_map=False, map_style='basic', showlegend=True, figsize=1,
                aspectratio='auto', filename='temp-plot.html'):
    """
    plots all the traces (which can be created using :func:`create_bus_trace`, :func:`create_line_trace`,
    :func:`create_trafo_trace`)
    to PLOTLY (see https://plot.ly/python/)

    INPUT:
        **traces** - list of dicts which correspond to plotly traces
        generated using: `create_bus_trace`, `create_line_trace`, `create_trafo_trace`

    OPTIONAL:
        **on_map** (bool, False) - enables using mapbox plot in plotly

        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **showlegend** (bool, 'True') - enables legend display

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the
            network geodata any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **filename** (str, "temp-plot.html") - plots to a html file called filename

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

    """

    if on_map:
        try:
            on_map = _on_map_test(traces[0]['x'][0], traces[0]['y'][0])
        except:
            logger.warning("Test if geo-data are in lat/long cannot be performed using geopy -> "
                           "eventual plot errors are possible.")

        if on_map is False:
            logger.warning("Existing geodata are not real lat/lon geographical coordinates. -> "
                           "plot on maps is not possible.\n"
                           "Use geo_data_to_latlong(net, projection) to transform geodata from specific projection.")

    if on_map:
        # change traces for mapbox
        # change trace_type to scattermapbox and rename x to lat and y to lon
        for trace in traces:
            trace['lat'] = trace.pop('x')
            trace['lon'] = trace.pop('y')
            trace['type'] = 'scattermapbox'
            if "line" in trace and isinstance(trace["line"], Line):
                # scattermapboxplot lines do not support dash for some reason, make it a red line instead
                if "dash" in trace["line"]._props:
                    _prps = dict(trace["line"]._props)
                    _prps.pop("dash", None)
                    _prps["color"] = "red"
                    trace["line"] = scmLine(_prps)
                else:
                    trace["line"] = scmLine(dict(trace["line"]._props))
            elif "marker" in trace and isinstance(trace["marker"], Marker):
                trace["marker"] = scmMarker(trace["marker"]._props)

    # setting Figure object
    fig = Figure(data=traces,  # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=showlegend,
                     autosize=(aspectratio == 'auto'),
                     hovermode='closest',
                     margin=dict(b=5, l=5, r=5, t=5),
                     # annotations=[dict(
                     #     text="",
                     #     showarrow=False,
                     #     xref="paper", yref="paper",
                     #     x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False),
                     # legend=dict(x=0, y=1.0)
                 ), )

    # check if geodata are real geographical lat/lon coordinates using geopy

    if on_map:
        try:
            mapbox_access_token = _get_mapbox_token()
        except Exception:
            logger.exception('mapbox token required for map plots. '
                             'Get Mapbox token by signing in to https://www.mapbox.com/.\n'
                             'After getting a token, set it to pandapower using:\n'
                             'pandapower.plotting.plotly.mapbox_plot.set_mapbox_token(\'<token>\')')
            raise MapboxTokenMissing

        fig['layout']['mapbox'] = dict(accesstoken=mapbox_access_token,
                                       bearing=0,
                                       center=dict(lat=pd.Series(traces[0]['lat']).dropna().mean(),
                                                   lon=pd.Series(traces[0]['lon']).dropna().mean()),
                                       style=map_style,
                                       pitch=0,
                                       zoom=11)

    # default aspectratio: if on_map use auto, else use 'original'
    aspectratio = 'original' if not on_map and aspectratio == 'auto' else aspectratio

    if aspectratio != 'auto':
        if aspectratio == 'original':
            # TODO improve this workaround for getting original aspectratio
            xs = []
            ys = []
            for trace in traces:
                xs += trace['x']
                ys += trace['y']
            x_dropna = pd.Series(xs).dropna()
            y_dropna = pd.Series(ys).dropna()
            xrange = x_dropna.max() - x_dropna.min()
            yrange = y_dropna.max() - y_dropna.min()
            ratio = xrange / yrange
            if ratio < 1:
                aspectratio = (ratio, 1.)
            else:
                aspectratio = (1., 1 / ratio)

        aspectratio = np.array(aspectratio) / max(aspectratio)
        fig['layout']['width'], fig['layout']['height'] = ([ar * figsize * 700 for ar in aspectratio])

    # check if called from ipynb or not in order to consider appropriate plot function
    if _in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot

    # delete the plot function here.
    # plot(fig, filename=filename)

    return fig

# Generating the plot for GENERAL users of the map

def pf_res_plotly_gen(net, capacity_limit, cmap="binary_r", use_line_geodata=None, on_map=False, projection=None,
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
            net.bus.name.astype(str) + '<br />'
            #'V_m = ' + net.res_bus.vm_pu.round(precision).astype(str) + 'pu' + '<br />' +
            #'V_m = ' + (net.res_bus.vm_pu * net.bus.vn_kv.round(2)).round(precision).astype(str) + ' kV' + '<br />' +
            #'V_a = ' + net.res_bus.va_degree.round(precision).astype(str)' +
            'Max_cap =' + net.bus.max_load.astype(str) + 'MW' + '<br />' +
            'cost =' + net.bus.cost.astype(str) + 'SEK/MWh' + '<br />').tolist()
            
    hoverinfo = pd.Series(index=net.bus.index, data=hoverinfo)

    bus_trace_less = create_bus_trace(net, net.bus[net.bus['max_load']<=capacity_limit].index, size=bus_size, infofunc=hoverinfo, color='red')
    bus_trace_more = create_bus_trace(net, net.bus[net.bus['max_load']>capacity_limit].index, size=bus_size, infofunc=hoverinfo, color='blue')

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    # if bus geodata is available, but no line geodata
    cmap_lines = 'binary' if cmap == 'binary' else cmap
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False
    # hoverinfo which contains name and pf results
    hoverinfo = (net.line.name.astype(str) + '<br />').tolist()
            #'I = ' + net.res_line.loading_percent.round(precision).astype(str) + ' %' + '<br />' +
            #'I_from = ' + net.res_line.i_from_ka.round(precision).astype(str) + ' kA' + '<br />' +
            #'I_to = ' + net.res_line.i_to_ka.round(precision).astype(str) + ' kA' + '<br />'
            
    hoverinfo = pd.Series(index=net.line.index, data=hoverinfo)
    line_traces = create_line_trace(net, use_line_geodata=use_line_geodata, respect_switches=True,
                                    width=line_width,
                                    infofunc=hoverinfo,
                                    cmap=cmap_lines,
                                    cmap_vals=net.res_line['loading_percent'].values,
                                    cmin=climits_load[1],
                                    cmax=climits_load[1])
                                    #cbar_title='Line Loading [%]',
                                    #cpos=cpos_load)
 
    # ----- Trafos ------
    # hoverinfo which contains name and pf results
    hoverinfo = (net.trafo.name.astype(str) + '<br />').tolist()
            #'I = ' + net.res_trafo.loading_percent.round(precision).astype(str) + ' %' + '<br />' +
            #'I_hv = ' + net.res_trafo.i_hv_ka.round(precision).astype(str) + ' kA' + '<br />' +
            #'I_lv = ' + net.res_trafo.i_lv_ka.round(precision).astype(str) + ' kA' + '<br />'
  
    hoverinfo = pd.Series(index=net.trafo.index, data=hoverinfo)
    trafo_traces = create_trafo_trace(net, width=line_width * 1.5, infofunc=hoverinfo,
                                      cmap=cmap_lines, cmin=0, cmax=100)
 
    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color='grey', size=bus_size * 2, trace_name='external_grid',
                                      patch_type=marker_type)
 
    return draw_traces_nograph(line_traces + trafo_traces + ext_grid_trace + bus_trace_less + bus_trace_more,
                       showlegend=False, aspectratio=aspectratio, on_map=on_map,
                       map_style=map_style, figsize=figsize, filename=filename)

# Generating map for the technical users

def pf_res_plotly_eng(net, cmap="Jet", use_line_geodata=None, on_map=False, projection=None,
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
    for geo_type in ["bus_geodata", "line_geodata"]:
        dupl_geo_idx = pd.Series(net[geo_type].index)[pd.Series(
                net[geo_type].index).duplicated()]
        if len(dupl_geo_idx):
            if len(dupl_geo_idx) > 20:
                logger.warning("In net.%s are %i duplicated " % (geo_type, len(dupl_geo_idx)) +
                               "indices. That can cause troubles for draw_traces()")
            else:
                logger.warning("In net.%s are the following duplicated " % geo_type +
                               "indices. That can cause troubles for draw_traces(): " + str(
                               dupl_geo_idx))


    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # ----- Buses ------
    # initializating bus trace
    # hoverinfo which contains name and pf results
    precision = 3
    hoverinfo = (
            net.bus.name.astype(str) + '<br />' +
            'V_m = ' + net.res_bus.vm_pu.round(precision).astype(str) + ' pu' + '<br />' +
            'V_m = ' + (net.res_bus.vm_pu * net.bus.vn_kv.round(2)).round(precision).astype(str) + ' kV' + '<br />' +
            'V_a = ' + net.res_bus.va_degree.round(precision).astype(str) + ' deg'
            'Max_cap =' + net.bus.max_load.astype(str) + 'MW' + '<br />' +
            'cost =' + net.bus.cost.astype(str) + 'SEK/MWh' + '<br />').tolist()

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

    return draw_traces_nograph(line_traces + trafo_traces + ext_grid_trace + bus_trace,
                       showlegend=False, aspectratio=aspectratio, on_map=on_map,
                       map_style=map_style, figsize=figsize, filename=filename)

# In[3]:



""" 

Extract data from time-series calculation. With result[0] being the vm_pu, result[1] being line_load, result[2] being trafo_loading_percent

Return two list, networks & figures. Which will be later used for Dash.

 Some points here: 
1. the time series loop through columns instead of rows. This is dealt in the two loop structure down below.

"""
def generate_graph_data_eng(net):
    net = net
    result = pfa.load_files()

    # create lists for time-series data, networks for net, figures for fig
    networks = [None]*len(result[0])
    figures = [None]*len(result[0])

    # looping to get the time-series data into the list ( i = number(timeseries) networks)
    for i in np.arange(len(result[0])):
        
        # here iterate through res_bus.vm_pu, ( result[0] )
        for j in np.arange(len(result[0].columns)):
            net.res_bus.vm_pu[j] = result[0][j][i]
        
        # here iterate through res_line.loading_percent, ( result[1] )
        for k in np.arange(len(result[1].columns)):
            net.res_line.loading_percent[k] = result[1][k][i]
        
        # here iterate through res_trafo.loading_percent, ( result[2] )
        for l in np.arange(len(result[2].columns)):
            net.res_trafo.loading_percent[l] = result[2][l][i]
        
        networks[i] = deepcopy(net)

    #looping to get the time series graph
    for ii in np.arange(len(networks)):
        figures[ii] = deepcopy(pf_res_plotly_eng(networks[ii],map_style='dark'))
       
    return networks, figures

""" 

Selecting the most important two graph for general users. 
The function needs to be " revised " later on for one more function to determine what is the graph we are going to show.

"""
def generate_graph_data_gen(networks_eng, capacity_limit):
    networks = networks_eng
    figures = [None]*2

    figures[0] = deepcopy(pf_res_plotly_gen(networks[5],capacity_limit,map_style='dark'))
    ''' Here the pf_res_plotly_eng is only for demo, it should be pf_res_plotly_gen just like the previous line '''
    figures[1] = deepcopy(pf_res_plotly_gen(networks[90],capacity_limit,map_style='dark'))
       
    return figures


# ### 2. Generate the grpah with Dash

# In[4]:


"""

Generate Dash graph. Calling this function along is enough for generating the grapgh.

Input:
should be with (net_data, time_step)

Outpuy:
return a link to the Dash site.

"""

def generate_graph(input_data, time):
    net = input_data
    time_steps = time
    
    networks = [None]*len(net.res_bus.vm_pu)
    figures = [None]*len(time_steps)
    # extract time-series values
    networks, figures = generate_graph_data(net)
    
    # take the correct order for slider
    list_length = len(networks)-1

    app = dash.Dash(__name__)

    # ------------------------------------------------------------------------------
    # App layout
    app.layout = html.Div([

        html.H1("Capacity Map with Dash component Testing", style={'text-align': 'center'}),

        dcc.Dropdown(id="slct_year",
                     options=[
                         {"label": "2015", "value": 2015},
                         {"label": "2016", "value": 2016},
                         {"label": "2017", "value": 2017},
                         {"label": "2018", "value": 2018}],
                     multi=False,
                     value=2015,
                     style={'width': "40%"}
                     ),

        html.Br(),

        dcc.Graph(id='my_powerFlow_graph',
                  style={
                      "margin-left": "auto",
                      "margin-right": "auto",
                  },
                  figure={}),
        html.Div(id='output_container_slider', children=[]),
        html.Br(),

        dcc.Slider(
            id='my-slider',
            min=0,
            max=list_length,
            step=1,
            value=1,
        ),

    ],
        # putting Style for the whole html.div block and it works!!!
    style={'width': '50%','padding-left':'25%', 'padding-right':'25%'},
    )


    # ------------------------------------------------------------------------------
    # Connect the Plotly graphs with Dash Components
    @app.callback(
        [Output(component_id='output_container_slider', component_property='children'),
         Output(component_id='my_powerFlow_graph', component_property='figure')],
        [Input(component_id='slct_year', component_property='value'),
         Input(component_id = 'my-slider',component_property='value')]
    )
    def update_graph(option_slctd, slider_slctd):
        print(option_slctd)
        print(type(option_slctd))

        container = "The year chosen by user was: {}".format(option_slctd)
        container_slider = "The time chosen by user was: {}".format(slider_slctd)

        fig_power = figures[slider_slctd]

        # 上面的output對應到這邊的return，是按照順序的
        # The output is correspoding to the return value below, by order
        return container_slider, fig_power


    # ------------------------------------------------------------------------------
    if __name__ == '__main__':
        return app.run_server(debug=True,use_reloader=False,port=3004)

