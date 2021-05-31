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
from pandapowermod.timeseries.run_time_series_mod import run_timeseries_mod
from pandapowermod.timeseries.run_time_series_mod1 import run_timeseries_mod1


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


def feas_chk_mod(net, conn_at_bus, loadorgen, size_p, size_q, prof):
    """
    Same as feas_chk but implements modified run_timeseries function of panadapower to skip unecessary steps.
    """
    init_all = get_init_all(net)
    net = add_loadgen(net, loadorgen, conn_at_bus, size_p, size_q, prof)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    sb.apply_const_controllers(net, profiles)  # create timeseries data from profiles and run powerflow
    chk = not run_timeseries_mod(net, time_steps, continue_on_divergence=True, verbose=True)  # Run powerflow only over time_steps
    net = init_net(net, init_all)
    return chk


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
        ul_chk = feas_chk_mod(net, ow, conn_at_bus, loadorgen, size_p=ul_p, size_q=inp_q, prof=prof)
        mid_chk = feas_chk_mod(net, ow, conn_at_bus, loadorgen, size_p=mid_p, size_q=inp_q, prof=prof)
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
    Iteratre the max_cap function over all busses in the grid.

    TODO: Add progess bar

    INPUT
        net (PP net) - Pandapower net
        ow (Object) - Output writer object
        loadorgen (str) - 'sgen' or 'load' for generation or load for additional capacity connected
        ll_p (int) - Size of maximum power limit of additional capacity that can be added
        ul_p (int) - Size of minimum additional capacity that can be added (Set as 0)
        prof (str) - Name of the profile. Must be available in the net.profiles of the input grid

    OUTPUT allcap (dataframe) - Maximum capacitiy of load/generation that can be added at all buses
    """
    len_items = len(net.bus)
    items = list(range(0, len_items))
    printProgressBar(0, len_items, prefix='Progress:', suffix='Complete', length=50)

    allcap = net.bus[['name', 'vn_kv']]
    allcap = allcap.join(pd.DataFrame(np.zeros([len(net.bus),len(net.line)])))
    for i, conn_at_bus in enumerate(items):
        for out_line in net.line.index:
            net.line.loc[out_line, "in_service"] = False
            allcap[out_line][conn_at_bus] = max_cap(net, ow=ow, conn_at_bus=conn_at_bus, loadorgen=loadorgen, ul_p=ul_p, ll_p=ll_p,
                                 prof=prof)
            net.line.loc[out_line, "in_service"] = True
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
    #return True
    feas_chk(net=net,ow=ow,conn_at_bus=conn_at_bus,loadorgen=loadorgen, size_p=size_p, size_q=size_q, prof=prof)
    return violations_long(net)



def resample_profiles(net, freq='H' ,head_values=96):
    """
    Resample profiles stored in net.profiles. Better alternative is resample_profiles_month

    INPUT
        net (PP net) - Pandapower net
        freq (str) - frequency of resampling
        head_values (int) - initial values to be resampled since the net.profiles is too large

    OUTPUT
        net (PP net) - Updated Pandapower net

    """
    for elm in net.profiles.keys():
        net.profiles[elm] = net.profiles[elm].head(head_values)
        net.profiles[elm].index = pd.to_datetime(
            net.profiles[elm].time)  # pd.date_range(start='1/1/2021',freq='H',periods=len(net.profiles['load']))
        # net.profiles[elm].drop('time', axis=1, inplace=True)
        if all(net.profiles[elm].columns == 'time'):
            net.profiles[elm] = net.profiles[elm].resample(freq).sum()
        else:
            net.profiles[elm] = net.profiles[elm].resample(freq).mean()
    return net

def resample_profiles_months(net, month=6):
    """
    Resample profiles stored in net.profiles for a paritular month into hours of day.

    INPUT
        net (PP net) - Pandapower net
        month (int) - month number

    OUTPUT
        net (PP net) - Updated Pandapower net

    """
    for elm in net.profiles.keys():
        #net.profiles[elm] = net.profiles[elm].groupby()
        net.profiles[elm].index = pd.to_datetime(net.profiles[elm].time, dayfirst=True )
        # pd.date_range(start='1/1/2021',freq='H',periods=len(net.profiles['load']))
        # net.profiles[elm].drop('time', axis=1, inplace=True)
        net.profiles[elm] = net.profiles[elm][net.profiles[elm].index.month == month]
        if all(net.profiles[elm].columns == 'time'):
            net.profiles[elm] = net.profiles[elm].groupby(net.profiles[elm].index.hour).sum()
        else:
            net.profiles[elm] = net.profiles[elm].groupby(net.profiles[elm].index.hour).mean()
        #net.profiles[elm].index = pd.to_datetime(net.profiles[elm].index)


    return net

def get_profiles(net, new_cap_name, cap_steps=4000, cap_step_size=0.05, pf=0.95,resample_freq='H'):
    """
    Second method of calculating capacity map using updated run_time_series function of pandapower repository.
    It joins multiple profiles together and runs a single run_timeseries.

    INPUT
        net (PP net) - Pandapower net
        new_cap_name (str) - same as loadorgen. type of capacity to add at a bus.
        cap_steps (int) - number of profiles to be joined together
        cap_step_size (int) - increment of each cap step which are to be joined.
        pf (int) - power factor
        resample_freq (str) - resampling frequency given to resample function

    OUTPUT
        profiles (tuple) - profiles to be loaded before run_timeseries. (For application of pandapower controller)

    """
    length_of_profiles = len(net.profiles['load'])
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    # profiles=[profile[type]*0.05 for type in profiles]
    for type in profiles:
        profiles[type] = pd.concat([profiles[type]] * cap_steps)
        profiles[type].index = pd.date_range(profiles[type].index[0], periods=len(profiles[type]), freq=resample_freq)
    df = pd.Series(range(length_of_profiles * cap_steps))
    df1 = pd.Series(range(length_of_profiles))
    df_add = ((df - pd.concat([df1] * cap_steps, ignore_index=True)) * cap_step_size / length_of_profiles)
    if new_cap_name == 'load':
        df_add.index = profiles['load', 'p_mw'].index
        profiles['load', 'p_mw'].iloc[:, -1] *= df_add
        profiles['load', 'q_mw'].iloc[:, -1] *= df_add * ((1- pf*pf)**0.5)
    elif new_cap_name == 'sgen':
        df_add.index = profiles['sgen', 'p_mw'].index
        profiles['sgen', 'p_mw'].iloc[:, -1] *= df_add
    for elm in profiles:
        profiles[elm].reset_index(inplace=True, drop=True)
    return profiles


def max_cap_map_new(net, loadorgen, conn_at_bus, prof, cap_steps=4000, cap_step_size=0.05, resample_freq='H', head_values_profiles=96):
    """
    Same as max_cap but works on the second method which joins same profiles together and runs a single timeseries instead of nested loops as in original method.
    Each profile joined has increased incremental capacity of cap_step_size on the conn_at_bus
    The single timeseries gets interrupted when violation occurs.
    Inputs/output similar to get_profiles and max_cap


    """
    elm_day = pd.Timedelta('1D') / pd.Timedelta(str(1) + resample_freq)
    init_all = get_init_all(net)
    net = add_loadgen(net, loadorgen, conn_at_bus=conn_at_bus, prof=prof, size_p=1, size_q=1)
    #net=resample_profiles(net, freq=resample_freq, head_values=head_values_profiles)
    profiles = get_profiles(net, new_cap_name=loadorgen, cap_steps=cap_steps, cap_step_size=cap_step_size, resample_freq=resample_freq)
    sb.apply_const_controllers(net, profiles)  # create timeseries data from profiles and run powerflow
    violation_at_step = run_timeseries_mod(net, continue_on_divergence=False,
                                          verbose=True)  # Run powerflow only over time_steps
    net=init_net(net, init_all)
    max_calc = (violation_at_step - violation_at_step % elm_day) * cap_step_size / elm_day
    return max_calc


def max_cap_map_new_cont(net, loadorgen, conn_at_bus, prof, cap_steps=100, cap_step_size=0.1, resample_freq='H', head_values_profiles=96):
    """
    Same as max_cap_map_new with contringency analysis.
    """
    elm_day = pd.Timedelta('1D') / pd.Timedelta(str(1) + resample_freq)
    min_violation_step=elm_day*cap_steps
    for out_line in net.line.index:
        net.line.loc[out_line, "in_service"] = False
        init_all = get_init_all(net)
        net = add_loadgen(net, loadorgen, conn_at_bus=conn_at_bus, prof=prof, size_p=1, size_q=1)
        #net=resample_profiles(net, freq=resample_freq, head_values=head_values_profiles)
        profiles = get_profiles(net, new_cap_name=loadorgen, cap_steps=cap_steps, cap_step_size=cap_step_size, resample_freq=resample_freq)
        sb.apply_const_controllers(net, profiles)  # create timeseries data from profiles and run powerflow
        violation_at_step = run_timeseries_mod(net, continue_on_divergence=False,
                                              verbose=True)  # Run powerflow only over time_steps
        net = init_net(net, init_all)
        if violation_at_step < min_violation_step:
            min_violation_step = violation_at_step
        net.line.loc[out_line, "in_service"] = True
    return (min_violation_step - min_violation_step % elm_day) * cap_step_size / elm_day


def all_cap_new(net,loadorgen,prof):
    """
    Looping max_cap_map_new_cont over all bus
    """
    allcap = net.bus[['name', 'vn_kv']]
    col_name = 'max_add_'+loadorgen
    allcap[col_name] = np.nan
    for i in range(97, len(net.bus)):
        allcap[col_name][i] = max_cap_map_new_cont(net, conn_at_bus=i, loadorgen=loadorgen, prof=prof)
    return allcap



ll_p = 0
ul_p = 90
inp_q = 0.1
s_tol = 0.005
time_steps = range(24)

output_dir = os.path.join(tempfile.gettempdir(), "simp_cap_v3")
# output_dir = os.path.join('C:\\Users\\nitbh\\OneDrive\\Documents\\IIPNB', "simp_cap_v3")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
