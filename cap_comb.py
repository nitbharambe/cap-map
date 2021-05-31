from cap import pfa
from cap import viz
import simbench as sb
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

ow = pfa.define_log(net, time_steps)  # For logging variables

## Get Map / PFA
'''
loadorgen: eg. 'load' , 'sgen'
prof: eg. 'L0-A' , 'WP4'
all_cap_map function takes lot of time to calculate capacities of an entire map. So they will be stored in an external file and read again in next section. Try using other inner functions to check if they work
Functions for finding maximum capacities
The all_cap_map function is max_cap looped for all busses so try below functions for quick results.
sing_res function is for checking individual case analysis
'''

net = pfa.resample_profiles_months(net, month=6)
sgen_allcap=pfa.all_cap_map(net,ow=ow, loadorgen='sgen', ul_p=ul_p, ll_p=ll_p, prof='WP4')
# load_allcap=pfa.all_cap_map(net,ow=ow, loadorgen='load', ul_p=ul_p, ll_p=ll_p, prof='L0-A')
# One sample run is needed to initiate visualisation
#pfa.max_cap(net,ow=ow,conn_at_bus=92, loadorgen='sgen',ul_p=ul_p, ll_p=ll_p, prof='WP4')
# pfa.sing_res(net,ow=ow,conn_at_bus=95, loadorgen='load',size_p=10,size_q=0.1, prof='L0-A')


## Visualisation

'''
Load data from all cap here. Calculated and stored earlier for saving time  
'''
# net.load['max_load']=pd.read_csv("sampdata/samp_load_allcap.csv")['max_add_cap']
net.sgen['max_sgen'] = pd.read_csv("sampdata/samp_sgen_allcap.csv")['max_add_cap']


#####################################################################
######################################################################
# Following section requires executing any one timeseries case (to be used with pfa.sing_res later) above to generate the graph. (Already done)

# extract time-series values
networks_eng, figures_eng = viz.generate_graph_data_eng(net)
figures_gen = viz.generate_graph_data_gen(networks_eng, 40)

# take the correct order for slider
list_length = len(networks_eng) - 1
# activate Dash
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# Dash component
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
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=0),
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

# the enginner part of the graph
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

# the place to set up layout
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
        # this is part is the engineer par
        engineer_part,
    ],
    # fluid=True,
)


# callback functions can be align with multiple callbacks
@app.callback(
    Output("cluster-graph", "figure"),
    [
        Input("capacity", "value"),
        Input("location", "value"),
        #Input("cluster-count", "value"),
        Input("summer_button", "n_clicks"),
        Input("winter_button", "n_clicks"),
    ],
)
# Here insert the graph from pandapower, for beginner part
def change_graph_input(x, y, summer_click, winter_click):
    container = "The year chosen by user was: {}".format(x)
    container_slider = "The time interval chosen by user was: {}".format(y)
    ctx = dash.callback_context
    count = ctx.triggered[0]['prop_id'].split('.')[0]
    if count == 'summer_button':
        return figures_gen[0]
    elif count == 'winter_button':
        return figures_eng[1]
    return figures_gen[0]

# From the following, the code is used for the second graph

# Connect the Plotly graphs with Dash Components, Fro expert part
@app.callback(
    [Output(component_id='output_container_slider', component_property='children'),
     Output(component_id='my_powerFlow_graph', component_property='figure')],
    [
    #    Input(component_id='slct_year', component_property='value'),
        Input(component_id='my-slider', component_property='value')
    ]
)
def update_graph(slider_slctd):

    #container = "The year chosen by user was: {}".format(option_slctd)
    container_slider = "The time interval chosen by user was: {}".format(slider_slctd)

    fig_power = figures_eng[slider_slctd]

    # 上面的output對應到這邊的return，是按照順序的
    # The output is correspoding to the return value below, by order
    return container_slider, fig_power


if __name__ == "__main__":
    app.run_server(debug=False, port=3004)