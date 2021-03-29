from cap import pfa
from cap import viz
import simbench as sb
import numpy as np
import pandas as pd
# for Dash import (might change Dash into a single function)
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

### Input Grid

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

### Other parameters

'''
Might put these inside .py file
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
ll_p=0
ul_p=90
inp_q=0.1
s_tol=0.005

ow=pfa.define_log(net,time_steps)   #For logging variables

### Input Parameters
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

## Get Map / PFA
'''
all_cap_map function takes lot of time to calculate capacities of an entire map. So they will be stored in an external file and read again in next section. Try using other inner functions to check if they work
'''

#Functions for finding maximum capacities
#sgen_allcap=pfa.all_cap_map(net,ow=ow, loadorgen='sgen', ul_p=ul_p, ll_p=ll_p, prof='WP4')
#load_allcap=pfa.all_cap_map(net,ow=ow, loadorgen='load', ul_p=ul_p, ll_p=ll_p, prof='L0-A')

#The all_cap_map function is max_cap looped for all busses so try below functions for quick results.
pfa.max_cap(net,ow=ow,conn_at_bus=92, loadorgen='sgen',ul_p=ul_p, ll_p=ll_p, prof='WP4')
#pfa.max_cap(net,ow=ow,conn_at_bus=95, loadorgen='load',ul_p=ul_p, ll_p=ll_p, prof='L0-A')
#pfa.sing_res(net,ow=ow,conn_at_bus=95, loadorgen='load',size_p=10,size_q=0.1, prof='L0-A')

#%% md

## Visualisation

#%%

net.load['max_load']=pd.read_csv("sampdata/samp_load_allcap.csv")['max_add_cap']
net.sgen['max_sgen']=pd.read_csv("sampdata/samp_sgen_allcap.csv")['max_add_cap']
# Or just initialize to random values
#net.sgen['max_sgen']=np.random.randint(0,100,net.sgen.shape[0])
#net.load['max_load']=np.random.randint(0,100,net.load.shape[0])


#####################################################################
######################################################################
# Required excuting pfa.max_cap() above to generate the graph

# extract time-series values
networks, figures = viz.generate_graph_data(net)

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


if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False,port=3004)