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
init_cap_l=[len(net.load) , len(net.sgen)]

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
ll_p=0
ul_p=90
inp_q=0.1
s_tol=0.005

#output_dir = os.path.join(tempfile.gettempdir(), "simp_cap_v3")
output_dir = os.path.join('C:\\Users\\nitbh\\OneDrive\\Documents\\IIPNB', "simp_cap_v3")
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


sgen_allcap=all_cap_map(net,ow=ow, loadorgen='sgen', ul_p=ul_p, ll_p=ll_p, prof='WP4')
