"""
Code to be refined a lot later on
TODO: Add default values of all functions
"""

# Import the pandapower and the networks module:
import pandapower as pp
import simbench as sb
import pandas as pd
import numpy as np
import os
import tempfile
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries


def get_init_all(net):
    """
    Returns initialization data for net
    INPUT:
        net : pandapower net
    OUTPUT:
        tuple of initial load, initial generation
    """
    initload = net.load[['p_mw', 'q_mvar']]
    initsgen = net.sgen[['p_mw', 'q_mvar']]
    return initload, initsgen


def init_net(net, init_all):
    """
    Drops any added load/generation Initialization of load and generation p_mw and q_mvar is needed because the
    run_timeseries replaces them after every iteration Drops Constcontrol objects created by sb.apply_cost_controllers

    OUTPUT -
        net- Pandapower network with initial values
    """
    [initload, initsgen] = init_all
    net.load = net.load.head(len(initload))
    net.sgen = net.sgen.head(len(initsgen))
    net.load[['p_mw', 'q_mvar']] = initload
    net.sgen[['p_mw', 'q_mvar']] = initsgen
    net.controller = net.controller.iloc[0:0]
    return net


def define_log(net, time_steps):
    """
    Creates output writer object required for timeseries simulation
    The timeseries module only calculates the values of variables mentioned here for each simulation.
    The temporary data gets stored in the output_dir directory

    OUTPUT
        ow - Output writer object
    """
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'loading_percent')
    return ow


def add_loadgen(net_t, loadorgen, conn_at_bus, size_p, size_q, prof):
    """
    Adds a load or generation to the net.load or net.sgen table.
    Adds profile name to the profiles variable of the newly addded capacity.

    INPUT
        net_t (PP net) - Pandapower net
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        conn_at_bus (int) - Bus at which additional capacity is connected
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid

    OUTPUT
        net_t (PP net) - Updated Pandapower net

    """
    if loadorgen == "load":
        pp.create_load(net_t, conn_at_bus, p_mw=size_p, q_mvar=size_q)
        net_t.load.tail(1).profile = prof
    elif loadorgen == "sgen":
        pp.create_sgen(net_t, conn_at_bus, p_mw=size_p, q_mvar=size_q)
        net_t.sgen.tail(1).profile = prof
    else:
        return 0
    return net_t


def load_files():
    """
    Loads files of previous TS simulation

    OUTPUT
        vm_pu,line_load,trafo_load (tuple) - Previous results of timeseries
    """
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.json")
    vm_pu = pd.read_json(vm_pu_file)
    line_load_file = os.path.join(output_dir, "res_line", "loading_percent.json")
    line_load = pd.read_json(line_load_file)
    trafo_load_file = os.path.join(output_dir, "res_trafo", "loading_percent.json")
    trafo_load = pd.read_json(trafo_load_file)
    return vm_pu, line_load, trafo_load


def violations_long(net):
    """
    Checks for any violations created in the grid by additional capacity and returns tuple with the details Loads the
    files created by timeseries simulation. Compares simulation values against the limits mentioned in the input grid.

    INPUT
        net (PP net) - Pandapower net

    OUTPUT
        check (bool) - tuple of violations with details

    """
    [vm_pu, line_load, trafo_load] = load_files()

    pf_vm_extremes = pd.DataFrame(vm_pu.max())
    pf_vm_extremes.columns = ['pf_max_vm_pu']
    pf_vm_extremes['pf_min_vm_pu'] = vm_pu.min()
    vm_pu_check = net.bus[['name', 'vn_kv', 'min_vm_pu', 'max_vm_pu']].join(pf_vm_extremes)
    vm_pu_check = vm_pu_check[
        (vm_pu_check.pf_max_vm_pu > vm_pu_check.max_vm_pu) | (vm_pu_check.pf_min_vm_pu < vm_pu_check.min_vm_pu)]

    pf_line_extremes = pd.DataFrame(line_load.max())
    pf_line_extremes.columns = ['pf_max_loading_percent']
    line_load_check = net.line[['name', 'from_bus', 'to_bus', 'max_loading_percent']].join(pf_line_extremes)
    line_load_check = line_load_check[(line_load_check.pf_max_loading_percent > line_load_check.max_loading_percent)]

    pf_trafo_extremes = pd.DataFrame(trafo_load.max())
    pf_trafo_extremes.columns = ['pf_max_loading_percent']
    trafo_load_check = net.trafo[['name', 'sn_mva', 'max_loading_percent']].join(pf_trafo_extremes)
    trafo_load_check = trafo_load_check[
        (trafo_load_check.pf_max_loading_percent > trafo_load_check.max_loading_percent)]

    return vm_pu_check, line_load_check, trafo_load_check


def violations(net):
    """
    Checks for any violations created in the grid by additional capacity. Loads the files created by timeseries
    simulation. Compares simulation values against the limits mentioned in the input grid.

    INPUT
        net (PP net) - Pandapower net

    OUTPUT
        check (bool) - 'True' for no violations. 'False' for violations present

    """
    [vm_pu, line_load, trafo_load] = load_files()

    check = any(np.where(vm_pu.max() > net.bus['max_vm_pu'], True, False))
    check = check or any(np.where(vm_pu.min() < net.bus['min_vm_pu'], True, False))
    check = check or any(np.where(line_load.max() > net.line['max_loading_percent'], True, False))
    check = check or any(np.where(trafo_load.max() > net.trafo['max_loading_percent'], True, False))
    return not check


def feas_chk(net, ow, conn_at_bus, loadorgen, size_p, size_q, prof):
    """
    Initializes the PPnet,
    Adds additional capacity,
    applies load/generation profiles on all the grid elements,
    runs timeseries for the specific case and save the results in the temporary output directory,
    Checks for violations

    TODO: Need to check process of how profiles from simbench are actually getting applied to Constcontrol know the
        fix. Also will lead to finding how profiles from input will be applied on the input grid.
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

    """
    init_all = get_init_all(net)
    net = add_loadgen(net, loadorgen, conn_at_bus, size_p, size_q, prof)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    sb.apply_const_controllers(net, profiles)  # create timeseries data from profiles and run powerflow
    run_timeseries(net, time_steps, continue_on_divergence=True, verbose=True)  # Run powerflow only over time_steps
    feas_result = violations(net)
    net = init_net(net, init_all)
    return feas_result


def max_cap(net, ow, conn_at_bus, loadorgen, ul_p, ll_p, prof):
    """
    Seach algorithm using feas_chk function over the range of ll_p and ul_p capacities

    TODO: Speed up, if it is required, try changing ul_p and ll_p as per voltage levels

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

    """
    no_iter = 0
    [ul_chk, mid_chk, ll_chk] = False, False, False
    while not (((ul_p - ll_p) < s_tol) | (ul_chk & mid_chk) | (no_iter > 7)):
        no_iter = no_iter + 1
        mid_p = (ul_p + ll_p) / 2
        ul_chk = feas_chk(net, ow, conn_at_bus, loadorgen, size_p=ul_p, size_q=inp_q, prof=prof)
        mid_chk = feas_chk(net, ow, conn_at_bus, loadorgen, size_p=mid_p, size_q=inp_q, prof=prof)
        if mid_chk:
            ll_p = mid_p
        elif not mid_chk:
            ul_p = mid_p
        elif ul_chk:
            return ul_p
    return ll_p


'''
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
'''


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar.
    the code is mentioned in : https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def all_cap_map(net, ow, loadorgen, ul_p, ll_p, prof):
    """
    Iteratre the max_cap function over all busses in the grid. Also sing_res for a slightly higher capacity to find
    out the limiting element

    TODO: Add progess bar

    INPUT
        net (PP net) - Pandapower net
        ow (Object) - Output writer object
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        ll_p (int) - Size of maximum power limit of additional capacity that can be added
        ul_p (int) - Size of minimum additional capacity that can be added (Set as 0)
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid

    OUTPUT allcap (dataframe) - Maximum capacitiy of load/generation that can be added at all buses and limiting
    element at each bus

    """
    len_items = len(net.bus)
    items = list(range(0, len_items))
    printProgressBar(0, len_items, prefix='Progress:', suffix='Complete', length=50)
    allcap = net.bus[['name', 'vn_kv']]
    allcap['max_add_cap'] = np.nan
    allcap['lim_elm'] = np.nan
    for i, conn_at_bus in enumerate(items):
        max_cap_at_bus = max_cap(net, ow=ow, conn_at_bus=conn_at_bus, loadorgen=loadorgen, ul_p=ul_p, ll_p=ll_p,
                                 prof=prof)
        allcap['max_add_cap'][conn_at_bus] = max_cap_at_bus
        allcap['lim_elm'][conn_at_bus] = sing_res(net, ow=ow, conn_at_bus=conn_at_bus, loadorgen=loadorgen,
                                                  size_p=max_cap_at_bus + 2 * s_tol, size_q=0.1, prof=prof)
        printProgressBar(i + 1, len_items, prefix='Progress:', suffix='Complete', length=50)
    return allcap


def sing_res(net, ow, conn_at_bus, loadorgen, size_p, size_q, prof):
    """
    Same as feas_chk. Only difference is that instead of returning bool it returns tuple with details of violations

    BUG: More like pending to do. Doesnt work for loads. Need to check process of how profiles from simbench are actually
    getting applied to Constcontrol know the fix. Also will lead to finding how profiles from input will be applied on
    the input grid.
    TODO: This function is incomplete. Hence returns 'True' for now.

    INPUT
        net (PP net) - Pandapower net
        ow (Object) - Output writer object
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        conn_at_bus (int) - Bus at which additional capacity is connected
        size_p (int) - Size of active power of additional capacity
        size_q (int) - Size of reactive power of additional capacity
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid

    OUTPUT
        result (tuple) - violations details

    """
    return True
    # feas_chk(net=net,ow=ow,conn_at_bus=conn_at_bus,loadorgen=loadorgen, size_p=size_p, size_q=size_q, prof=prof)
    # return violations_long(net)


ll_p = 0
ul_p = 90
inp_q = 0.1
s_tol = 0.005
time_steps = range(96)

output_dir = os.path.join(tempfile.gettempdir(), "simp_cap_v3")
# output_dir = os.path.join('C:\\Users\\nitbh\\OneDrive\\Documents\\IIPNB', "simp_cap_v3")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
