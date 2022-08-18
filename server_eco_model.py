"""_summary_
This experiment's configuration is defined fixed bits length that clients send to server every batch.
client size is increasing for each batch.
Returns:
    _type_: _description_
"""
from math import ceil, log, exp, sqrt, pi
import random
from typing import Dict, List
from xml.etree.ElementPath import find

import numpy as np

from privacy_module import PrivacyModule
from server import FAServerPEM

from utils import _init_clients, visualize_frequency


random.seed(0)

class FAServer(FAServerPEM):

    def client_valuation(self, type, x=0): 
        client_valuation_gaussian = lambda mu, sigma: 1/(sigma*sqrt(2*pi))*exp(-((x-mu)**2)/(2*sigma**2))
        client_valuation_uniform = lambda low, high: random.uniform(low, high)
    
        if type == "gaussian": 
            return client_valuation_gaussian
        elif type == "uniform":
            return client_valuation_uniform

    def simulate_client_profit(self):
        self.clients_privacy_budget = np.random.uniform(0.5, 3, self.n) # heterogeneous clients' privacy tolerance
        self.clients_privacy_cost = [max(-(self.clients_privacy_budget[i] - self.varepsilon), 0) + sqrt(self.varepsilon) for i in range(self.n)]
        client_valuation_func_type = "gaussian"
        client_valuation_func = self.client_valuation(client_valuation_func_type, self.varepsilon)
        if client_valuation_func_type == "gaussian":
            clients_means = self.clients_privacy_budget  # when server required privacy budget is close to the client's, the client receives higher valuation.
            clients_variances = [0.21] * self.n
            client_valuation_value = lambda a, mean, var: a*client_valuation_func(mean, var)  # `a` is magnification factor
            self.clients_valuation = [client_valuation_value(20, clients_means[i], clients_variances[i]) for i in range(self.n)] # valuations for each client
        else:
            pass
    
    def get_participants(self):
        """ Return clients that participate in the experiment. 

        Returns:
            _type_: clients who decide to join in
        """
        self.simulate_client_profit()
        return list(filter(lambda i: self.clients_valuation[i] > self.clients_privacy_cost[i], range(self.n)))

    def predict_heavy_hitters(self) -> Dict:
        """_summary_

        Args:
            n (int): client size
            m (int): binary-string length
            k (int): top-k heavy hitters
            varepsilon (float): privacy budget
            batch_size (int): number of groups
        Returns:
            Dict: top-k heavy hitters C_g and their frequencies.
        """
        
        # adder_base = ceil((2*self.n)/(self.batch_size*(self.batch_size+1)))

        bits_per_batch = ceil(self.m / self.batch_size)

        s_0 = 0
        
        C_i = {}
        C_i[0] = 0
        
        find_budget = False

        for i in range(self.batch_size):
            min_clients = 0.03*(i+1)*self.n
            
            self.varepsilon += 0.2

            participants = self.get_participants()


            while len(participants) == 0:
                self.varepsilon -= 0.1
                participants = self.get_participants()

                
            # while (not find_budget) and len(participants) < min_clients:
            #     self.varepsilon -= 0.1
            #     if self.varepsilon < 0:
            #         self.varepsilon = self.max_budget
            #         participants = self.get_participants()
            #         find_budget = True
            #         break
            #     participants = self.get_participants()


                
            print(f"Batch [{i}] :: Total {len(participants)} participants under Privacy Budget {self.varepsilon}")
            self.max_budget = self.varepsilon

            s_i = min(s_0 + bits_per_batch, self.m)
            delta_s = s_i - s_0
            s_0 = s_i

            D_i = {}
            for val in C_i.keys():
                for offset in range(2**delta_s):
                    D_i[(val << delta_s) + offset] = C_i[val]/(2**delta_s) # inherit weight_score

            
            # print("Privacy mechanism type:", self.privacy_mechanism_type)
            privacy_module = PrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type)
            # mechanism = privacy_mechanism(
            #     self.varepsilon, D_i, self.privacy_mechanism_type)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []

                        
            for client_idx in participants:
                client_data = self.clients[client_idx]
                prefix_client = client_data >> (self.m-s_i)
                response = mechanism(prefix_client)
                clients_responses.append(response)
     

            D_i = handle_response(clients_responses)

            D_i_sorted = sorted(D_i.items(), key=lambda x: x[-1], reverse=True)


            C_i = {}
            for indx in range(min(self.k, len(D_i_sorted))):
                v, count = D_i_sorted[indx]
                if count > 0:
                    C_i[v] = count
            # print(f"Group {i} generated: {C_i}")
        return C_i


if __name__ == '__main__':
    n = 1000

    m = 32
    k = 9
    init_varepsilon = 3.4
    step_varepsilon = 0.2
    max_varepsilon = 3 
    batch_size = 16

    sampling_rate = 1
    round = 1

    privacy_mechanism_type = "GRR_Weight" # ["GRR", "None","OUE"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]

    server = FAServer(n, m, k, init_varepsilon, batch_size, round, privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type, \
        sampling_rate= sampling_rate)
    
    # print(server.predict_heavy_hitters())
    # server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)
    server.server_run()

    # visualize_frequency(server.clients, server.C_truth, distribution_type=server.client_distribution_type)
