# %%
import random 
import numpy as np
import math
from collections import Counter
# add package path with current directory
import sys
sys.path.append('..')
from _utils import selectData, PruneHH
from PIRAPPOR import PIRAPPOR
from OHE_2RR import OHE_2RR
from typing import Dict, List
from server import BaseServer

class OptPrefixTreeServer(BaseServer):
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, 
        clients: List = [], C_truth: List = [], evaluate_type: str = "F1", 
        connection_loss_rate: float = 0, stop_iter = -1, FPR=0.5, tau_0=0.5, eta=0.2, denylist=[], l_t=-1):
        """ FPR (float): false positive ratio
            tau_0 (_type_): initialization of threshold
            eta (_type_): extra parameters for pruneHH
        """
        super().__init__(n, m, k, varepsilon, iterations, round, clients, C_truth, "None", evaluate_type, connection_loss_rate)
        
        if stop_iter == -1:
            self.stop_iter = iterations
        else: 
            self.stop_iter = stop_iter
        
        self.P = 1e7
        self.eta = eta
        self.tau_0 = tau_0
        self.FPR = FPR
        self.denylist = denylist
        self.l_t = l_t

    def ServerSide(self, V_t, PrivateAgg):
        """Server side algorithm per round

        Args:
            V_t (list): aggregated sum of devices response
            D_t (list): data domain of iteration t
            self.varepsilon (float): local privacy parameter
            PrivateAgg (function): private aggregation function
        """
        # print("debug:: len(V_t): ", len(V_t))
        f_est, sorted_D_t = PrivateAgg(V_t) # takes the aggregated privated responses and computes an estimatino of the frequency of every data element.
        # print("debug::D_t: ", D_t)
        # print("debug::f_est: ", f_est)
        
        sigma = np.sqrt(np.var(f_est)) # computes an upper bound on the standard deviation of the frequency estimate for d.
        
        P_prefixlist = PruneHH(sorted_D_t, f_est, self.tau_0, self.FPR, sigma, self.eta)
        if len(P_prefixlist) == 0: return P_prefixlist, 1
        l_t = int(math.log(self.P/len(P_prefixlist), 10)) # computes the optimal length of the prefix list for the next iteration.

        return P_prefixlist,l_t
    
    def predict_heavy_hitters(self):
       
        # %%
        # intialized segment length and prefix list

        l_t = 0
        l_pref = math.ceil(math.log(self.P, 10))
        print("Prefix length: ", l_pref)
        print("Segment length: ", l_t)
        P_prefix_t = []
        # P_discorvedlist = []

        for t in range(self.stop_iter):
            if t ==0: A_l_t = [a for a in range(2**l_pref)]
            else: A_l_t = [a for a in range(2**l_t)]
            
            D_t = [] # data domain for iteration 
            if len(P_prefix_t) == 0:
                P_prefix_t = A_l_t
                D_t = A_l_t
                # print("Round ", t, " Domain: ", P_prefix_t)
            else:
                for prefix in P_prefix_t:
                    prefixes = [(prefix << l_t) + a for a in A_l_t]
                    D_t = D_t + prefixes
                # print("Round ", t, " Domain: ", D_t)
                
            V_t = []
            private_machanism = OHE_2RR(D_t, self.varepsilon)
            for d in random.choices(self.clients, k=int(self.n)):
                if d in self.denylist: continue
                v_i = private_machanism.local_randomizer(d >> (self.m-(l_pref)))
                # v_i = DeviceSide(self.varepsilon, l_pref-l_t, l_t, P_prefix_t, P_denylist, selectData, private_machanism.local_randomizer)
                if v_i == None:
                    continue
                V_t.append(v_i)
                
            P_prefix_t, l_t = self.ServerSide(V_t,private_machanism.estimate_all_freqs) # server returns prefix list and segment length for next round
            if len(P_prefix_t) == 0: break
            if self.l_t != -1: l_t = self.l_t # fixed segment length
            else: l_t = max(1, l_t) 
            if l_pref >= self.m:
                break
            if l_pref + l_t > self.m:
                l_t = self.m - l_pref
            l_pref += l_t
            
        return {item: 0 for item in P_prefix_t}        



# # DeviceSide
# def DeviceSide(self.varepsilon, l_pref, l_t, P_prefix, P_denylist, selectData, local_randomizer):
#     """_summary_

#     Args:
#         self.varepsilon (_type_): local privacy parameter
#         l_pref (_type_): prefix length
#         l_t (_type_): segment length
#         P_prefix (_type_): allowed prefix list
#         P_denylist (_type_): deny list
#         selectData (_type_): function to choose a datapoint from the data.
#     """
#     D = [0b11110010, 0b11110010, 0b01010010]
#     D_pre = []
#     for d in D:
#         # prefix with length l_pref for d
#         prefix = d >> (r - l_pref)
#         if prefix in P_prefix and prefix not in P_denylist:
#             D_pre.append(d)
#     # print("debug:: D_pre: ", D_pre)
#     if len(D_pre) ==0:
#         d = None # reserve a special data element for users that have no eligible data points to report.
#         return d
#     else:
#         d = selectData(D_pre)
#     # print("debug:: d", d) 
#     v = local_randomizer(d >> (r-(l_pref+l_t)))
#     # print("debug:: v: ", v)
#     return v 