"""_summary_
This experiment's configuration is defined fixed bits length that clients send to server every batch.
client size is increasing for each batch.
Returns:
    _type_: _description_
"""
import pickle
from math import ceil, log
import random
from typing import Dict, List
from utils import load_clients, sort_by_frequency, visualize_frequency
from privacy_module import PrivacyModule
from server import FAServerPEM
import matplotlib.pyplot as plt
import numpy as np


random.seed(0)

class FedFTServer(FAServerPEM):

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
        stop_iter = 10


        # adder_base = int((2*self.n)/(self.iterations*(self.iterations+1)))
        adder_base = int((2*self.n)/((stop_iter*(stop_iter+1)))) 
        participants = 0 
        bits_per_batch = ceil(self.m / self.iterations)

        s_0 = 0
        A_i = {}
        A_i[0] = 0 # initial weight_score

     

        for i in range(self.iterations):
            if stop_iter-1 < i:
                print(self.m-s_0)
                return dict((key<<(self.m-s_0), value) for (key, value) in A_i.items())
            s_i = min(s_0 + bits_per_batch, self.m)
            delta_s = s_i - s_0
            s_0 = s_i

            C_i = {}
            for val in A_i.keys():
                for offset in range(2**delta_s):
                    C_i[(val << delta_s) + offset] = A_i[val]/(2**delta_s) if self.WT else 0 # inherit weight_score

            privacy_module = PrivacyModule(self.varepsilon, C_i, type=self.privacy_mechanism_type, batch=i+1, WT=self.WT, s_i = s_i)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []

            if self.is_uniform_size : adder = int(self.n/self.iterations)
            
            else: adder = (i+1)*adder_base
            print(f"Sampling {adder} clients")
            end_participants = participants + adder

            if i == stop_iter-1:
                end_participants = self.n

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
            for indx in range(min(self.k, len(C_i_sorted))):
                v, count = C_i_sorted[indx]
                if count > 0:
                    A_i[v] = count
            # print(f"Group {i} generated: {A_i}")
        return A_i


if __name__ == '__main__':
    n = 2000
    
    m = 20
    k = 1
    init_varepsilon = 12
    step_varepsilon = 0.5
    max_varepsilon = 9
    iterations = 10
    

    round = 10

    truth_top_k, clients = load_clients(filename="./dataset/synthetic_steps.txt", k=k, encode=False)
    # truth_top_k =[]
    # clients = []
    privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
    evaluate_module_type = "NDCG" # ["NDCG", "F1"]
    WT = True
    # ----Weight Tree & Client Size fitting---- # 
    server = FedFTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth = truth_top_k, \
        privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type, \
            WT = WT
        )

    # server.server_run_plot_varepsilon(init_varepsilon, step_varepsilon, max_varepsilon)
    # visualize_frequency(server.clients, server.C_truth, distribution_type=server.client_distribution_type)

    # ----------------------------------------------------------------------------
    HHs = list(server.predict_heavy_hitters().keys())
    print("Predict hhs:", HHs)

    with open("plural.txt", "wb") as f:
        pickle.dump([HHs, server.clients], f)

    


