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
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

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
net.bus.max_vm_pu = net.bus.max_vm_pu * 1.05

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
time_steps = range(96)
ll_p = 0
ul_p = 90
inp_q = 0.1
s_tol = 0.005

ow = pfa.define_log(net, time_steps)  # For logging variables

### Input Parameters
'''
Sample input. Use as needed
prof and loadorgen are needed for getting the map
Others are needed in case of individual capacity checks
'''
# size_pmw=10
# size_qmw=0.05
loadorgen = 'sgen'
# prof='L0-A'
prof = 'WP4'
# conn_at_bus=2

## Get Map / PFA
'''
all_cap_map function takes lot of time to calculate capacities of an entire map. So they will be stored in an external file and read again in next section. Try using other inner functions to check if they work
Functions for finding maximum capacities
The all_cap_map function is max_cap looped for all busses so try below functions for quick results.
sing_res function is for checking individual case analysis
'''

# sgen_allcap=pfa.all_cap_map(net,ow=ow, loadorgen='sgen', ul_p=ul_p, ll_p=ll_p, prof='WP4')
# load_allcap=pfa.all_cap_map(net,ow=ow, loadorgen='load', ul_p=ul_p, ll_p=ll_p, prof='L0-A')
''' To generate the data, need to remove the comment of next line of code '''
pfa.max_cap(net,ow=ow,conn_at_bus=92, loadorgen='sgen',ul_p=ul_p, ll_p=ll_p, prof='WP4')
# pfa.max_cap(net,ow=ow,conn_at_bus=95, loadorgen='load',ul_p=ul_p, ll_p=ll_p, prof='L0-A')
# pfa.sing_res(net,ow=ow,conn_at_bus=95, loadorgen='load',size_p=10,size_q=0.1, prof='L0-A')

## Visualisation

'''
Load data from all cap here. Calculated and stored earlier for saving time  
'''
# net.load['max_load']=pd.read_csv("sampdata/samp_load_allcap.csv")['max_add_cap']
net.sgen['max_sgen'] = pd.read_csv("sampdata/samp_sgen_allcap.csv")['max_add_cap']
# Or we can also just initialize to random values
# net.sgen['max_sgen']=np.random.randint(0,100,net.sgen.shape[0])
# net.load['max_load']=np.random.randint(0,100,net.load.shape[0])
net.bus['max_load'] = np.random.randint(0, 100, net.bus.shape[0])
net.bus['cost'] = np.random.randint(0, 100, net.bus.shape[0])

#####################################################################
######################################################################
'''
    viz.generate_graph_data_xxx extracts time-series data from the calculation above and align
them in a proper format for Dash.
'''
networks_eng, figures_eng = viz.generate_graph_data_eng(net)
figures_gen = viz.generate_graph_data_gen(networks_eng, 40)

list_length = len(networks_eng) - 1
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Step1: Input capacity"),
                html.Br(),
                dcc.Input(
                    id="capacity",
                    type="number",
                    placeholder="Enter a capacity (MW)",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Step2: Select location"),
                dcc.Dropdown(
                    id="location",
                    placeholder="Select a location",
                    options=[
                        {"label": "Stockholm", "value": "col"},
                        {"label": "Uppsala", "value": "col2"},
                        {"label": "Öland", "value": "col3"},
                    ],
                    value="sepal width (cm)",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Check the capacity in different time"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Summer", id="summer_button", active=True, color="success", className="mr-1"),
                            width={"size": 5},
                        ),
                        dbc.Col(
                            dbc.Button("Winter", id="winter_button", color="primary", className="mr-1"),
                            width={"size": 5},
                        ),
                    ]
                ),
            ]
        ),
    ],
    body=True,
)
engineer_part = html.Div(
    [
        html.H1("Capacity Map with Dash component, for Time-series usage", style={'text-align': 'center'}),

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
            value=0,
        ),
    ]
)
# setting up layout
app.layout = dbc.Container(
    [
        html.H1("Capacity Map with Dash component, for general usage", style={'text-align': 'center'}),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(controls, md=4, align="start"),
                dbc.Col(dcc.Graph(id="cluster-graph", figure={}), md=8),
            ],
            align="center",
        ),
        engineer_part,
    ],
)


@app.callback(
    Output("cluster-graph", "figure"),
    [
        Input("capacity", "value"),
        Input("location", "value"),
        Input("summer_button", "n_clicks"),
        Input("winter_button", "n_clicks"),
    ],
)
# Here insert the graph from pandapower, for general usage
def change_graph_input(x, y, summer_click, winter_click):
    ctx = dash.callback_context
    count = ctx.triggered[0]['prop_id'].split('.')[0]
    if count == 'summer_button':
        return figures_gen[0]
    elif count == 'winter_button':
        return figures_eng[1]
    return figures_gen[0]


# Here insert the graph from pandapower, for engineer usage
@app.callback(
    [Output(component_id='output_container_slider', component_property='children'),
     Output(component_id='my_powerFlow_graph', component_property='figure')],
    [
        Input(component_id='my-slider', component_property='value')
    ]
)
def update_graph(slider_slctd):
    container_slider = "The time interval chosen by user was: {}".format(slider_slctd)
    fig_power = figures_eng[slider_slctd]
    return container_slider, fig_power


if __name__ == "__main__":
    app.run_server(debug=False, port=3004)