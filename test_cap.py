from cap import pfa
from cap import viz
import simbench as sb
import numpy as np
import pandas as pd
#from pandapowermod.timeseries.run_time_series import run_timeseries_mod
#from pandapower.timeseries.run_time_series import run_timeseries

sb_code1 = "1-MV-rural--1-sw"  # rural MV grid of scenario 0 with full switchs
sb_code2 = "1-HVMV-urban-all-0-sw"  # urban hv grid with one connected mv grid which has the subnet 2.202
net = sb.get_simbench_net(sb_code1)
net.bus.max_vm_pu=net.bus.max_vm_pu*1.05

time_steps=range(96)
ll_p=0
ul_p=90
inp_q=0.1
s_tol=0.005

#ow=pfa.define_log(net,time_steps)   #For logging variables
size_pmw=1
#size_qmw=0.05
loadorgen='sgen'
#prof='L0-A'
prof='WP4'
#conn_at_bus=2

from cap import pfa
from cap import viz
import simbench as sb
import numpy as np
import pandas as pd
#from pandapowermod.timeseries.run_time_series import run_timeseries_mod
#from pandapower.timeseries.run_time_series import run_timeseries

sb_code1 = "1-MV-rural--1-sw"  # rural MV grid of scenario 0 with full switchs
sb_code2 = "1-HVMV-urban-all-0-sw"  # urban hv grid with one connected mv grid which has the subnet 2.202
net = sb.get_simbench_net(sb_code1)
net.bus.max_vm_pu=net.bus.max_vm_pu*1.05

time_steps=range(96)
ll_p=0
ul_p=90
inp_q=0.1
s_tol=0.005

#ow=pfa.define_log(net,time_steps)   #For logging variables
size_pmw=1
#size_qmw=0.05
loadorgen='sgen'
#prof='L0-A'
prof='WP4'
#conn_at_bus=2

net.load.iloc[30]['p_mw']=100
print(net.load.iloc[30]['p_mw'])
profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
sb.apply_const_controllers(net, profiles)  # create timeseries data from profiles and run powerflow
#violation_at_step = run_timeseries_mod(net, time_steps=range(5), continue_on_divergence=True, verbose=True)
#run_timeseries(net, time_steps=range(5), continue_on_divergence=True, verbose=True)
feas_chk_test(net, ow=ow, conn_at_bus=92, loadorgen='sgen', size_p=1, size_q=0.01, prof='WP4')
#violation_at_step