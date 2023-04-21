"""
FedFTServer will be updated to that in 'server_FAServer_bit.py' in the future.
"""
from math import ceil, log
import random
from typing import Dict, List

from privacy_module import PrivacyModule
from server import FAServerPEM
from _Tree import TrieNumeric
from utils import load_clients, blockPrint, enablePrint

blockPrint()
class Aserver(FAServerPEM):
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, 
        clients: List = [], C_truth: List = [], privacy_mechanism_type: List = "GRR_X", evaluate_type: str = "F1", 
        connection_loss_rate: float = 0, is_uniform_size: bool=False, stop_iter = -1):
            super().__init__(n, m, k, varepsilon, iterations, round, clients, C_truth, privacy_mechanism_type, evaluate_type, connection_loss_rate)
            self.bits_per_iter =ceil(self.m / self.iterations) 
            self.trie = TrieNumeric(self.bits_per_iter, k = k)
            self.is_uniform_size = is_uniform_size
            if stop_iter == -1:
                self.stop_iter = iterations
            else: 
                self.stop_iter = stop_iter
    def predict_heavy_hitters(self) -> Dict:
        """_summary_

        Args:
            n (int): client size
            m (int): binary-string length
            k (int): top-k heavy hitters
            varepsilon (float): privacy budget
            iterations (int): number of groups
        Returns:
            Dict: top-k heavy hitters C_g and their frequencies.
        """

        s_0 = 0
        A_i = {}
        A_i[0] = 0 # initial weight_score
        participants = 0
        bits_per_batch = ceil(self.m / self.iterations)

        for i in range(self.stop_iter):
            s_i = min(s_0 + bits_per_batch, self.m) # current bit length
            delta_s = s_i - s_0 # required bit length, e.g. 2bits per iter.
            s_0 = s_i # last iter's bit length

            privacy_module = PrivacyModule(self.varepsilon, A_i, type=self.privacy_mechanism_type,s_i = s_i, required_bits = delta_s)
            mechanism = privacy_module.privacy_mechanism()
            
            if privacy_module.p <= 0.5:
                privacy_module = PrivacyModule(self.varepsilon, A_i, type="GRR",s_i = s_i, required_bits = delta_s)
                mechanism = privacy_module.privacy_mechanism() 
                
            handle_response = privacy_module.handle_response() 
            clients_responses = []

            if self.is_uniform_size : adder = int(self.n/self.stop_iter)
            
            else: adder = int(self.n /(2*self.stop_iter) + (i) * self.n / (self.stop_iter* (self.stop_iter + 1)))
            print(f"Sampling {adder} clients")
            end_participants = participants + adder

            if i == self.stop_iter-1:
                end_participants = self.n-1

            for client in self.clients[participants: end_participants+1]:
                prefix_client = client >> (self.m-s_i)
                response = mechanism(prefix_client)
                p = random.random() 
                if p >= self.connection_loss_rate:
                    clients_responses.append(response)
            participants = end_participants

            C_i = handle_response(clients_responses)

            C_i_sorted = sorted(C_i.items(), key=lambda x: x[-1], reverse=True)


            A_i = {}
            # a  = self.k*2**self.bits_per_iter if i==self.stop_iter else self.k
            a = len(C_i_sorted) if i==self.stop_iter-1 else self.k
            for indx in range(min(a, len(C_i_sorted))):
                v, count = C_i_sorted[indx]
                if count > 0:
                    A_i[v] = 0  # validate v in next iteration
                    if i == self.stop_iter-1: A_i[v] = count
            if not A_i:
                A_i = {0: 0}
        self.accurate_bits = self.m-s_0
        return A_i
  