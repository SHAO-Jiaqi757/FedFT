# %%
import random 
import numpy as np
import math
from collections import Counter
# add package path with current directory
import sys
sys.path.append('..')
from utils import selectData, PruneHH
from PIRAPPOR import PIRAPPOR
from OHE_2RR import OHE_2RR
# DeviceSide
def DeviceSide(eps_l, l_pref, l_t, P_prefix, P_denylist, selectData, local_randomizer):
    """_summary_

    Args:
        eps_l (_type_): _description_
        l_pref (_type_): prefix length
        l_t (_type_): segment length
        P_prefix (_type_): allowed prefix list
        P_denylist (_type_): deny list
        selectData (_type_): function to choose a datapoint from the data.
    """
    D = [0b11110010, 0b11110010, 0b01010010]
    D_pre = []
    for d in D:
        # prefix with length l_pref for d
        prefix = d >> (r - l_pref)
        if prefix in P_prefix and prefix not in P_denylist:
            D_pre.append(d)
    # print("debug:: D_pre: ", D_pre)
    if len(D_pre) ==0:
        d = None # reserve a special data element for users that have no eligible data points to report.
        return d
    else:
        d = selectData(D_pre)
    # print("debug:: d", d) 
    v = local_randomizer(d >> (r-(l_pref+l_t)))
    # print("debug:: v: ", v)
    return v

# %%
def ServerSide(V_t, D_t, eps_l, FPR, tau_0, eta, PrivateAgg):
    """Server side algorithm per round

    Args:
        V_t (list): aggregated sum of devices response
        D_t (list): data domain of iteration t
        eps_l (float): local privacy parameter
        FPR (float): false positive ratio
        tau_0 (_type_): initialization of threshold
        eta (_type_): extra parameters for pruneHH
        PrivateAgg (function): private aggregation function
    """
    # print("debug:: len(V_t): ", len(V_t))
    f_est, sorted_D_t = PrivateAgg(V_t, D_t) # takes the aggregated privated responses and computes an estimatino of the frequency of every data element.
    # print("debug::D_t: ", D_t)
    # print("debug::f_est: ", f_est)
    
    sigma = np.sqrt(np.var(f_est)) # computes an upper bound on the standard deviation of the frequency estimate for d.
    
    P_prefixlist = PruneHH(sorted_D_t, f_est, tau_0, FPR, sigma, eta)
        
    l_t = int(math.log(P/len(P_prefixlist), 2)) # computes the optimal length of the prefix list for the next iteration.

    return P_prefixlist,l_t
    


# %%
A = [0, 1] # alphebet of the data
r =8  # fixed length of the data
N = 1000 # client number
T = 8 # number of iterations
eps_l = 1 # local privacy parameter
P = 10 # Bound on the dimension
tau_0 = 0.5
FPR = 0.1 # False positive ratio
eta = 0.2 # extra parameters to pass to ServerSide
P_denylist = [] # deny list

D = [i for i in range(2**r)] # data domain

# %%
# intialized segment length and prefix list

l_t = 0
l_pref = math.ceil(math.log(P))
print("Prefix length: ", l_pref)
print("Segment length: ", l_t)
P_prefix_t = []
# P_discorvedlist = []

for t in range(T):
    if l_pref > r:
        l_pref = r
        l_t = r - l_pref
    if t ==0: A_l_t = [a for a in range(2**l_pref)]
    else: A_l_t = [a for a in range(2**l_t)]
    
    D_t = [] # data domain for iteration 
    if len(P_prefix_t) == 0:
        P_prefix_t = A_l_t
        D_t = A_l_t
        print("Round ", t, " Domain: ", P_prefix_t)
    else:
        for prefix in P_prefix_t:
            prefixes = [(prefix << l_t) + a for a in A_l_t]
            D_t = D_t + prefixes
        print("Round ", t, " Domain: ", D_t)
        
    V_t = []
    # private_machanism = PIRAPPOR(2**(l_pref+l_t), eps_l)
    private_machanism = OHE_2RR(2**(l_pref+l_t), eps_l)
    global_raw_value = []
    for i in range(N):
        v_i = DeviceSide(eps_l, l_pref-l_t, l_t, P_prefix_t, P_denylist, selectData, private_machanism.local_randomizer)
        # v_i = random.randint(0, 2**l_t)
        if v_i == None:
            print("Device ", i, " has no eligible data")
            continue
        V_t.append(v_i)
        
    P_prefix_t, l_t = ServerSide(V_t, D_t, eps_l, FPR, tau_0, eta, private_machanism.estimate_all_freqs) # server returns prefix list and segment length for next round
    if len(P_prefix_t) == 0: break
    l_t = max(1, l_t)
    if l_pref >= r:
        break
    l_pref += l_t
    
    
print("Optimal Prefix Tree Algorithm Finished!")
print("Prefix list: ", P_prefix_t)
print("Denylist: ", P_denylist)
# print("Discorvedlist: ", P_discorvedlist)
    
        