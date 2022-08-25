"""_summary_
This experiment's configuration is defined fixed bits length that clients send to server every batch.
client size is increasing for each batch.
Returns:
    _type_: _description_
"""
from collections import defaultdict
from math import ceil, log
import random
from typing import Dict, List

from privacy_module import PrivacyModule
from server import FAServerPEM

from utils import visualize_frequency


random.seed(0)

class FAServerPreHashing(FAServerPEM):
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
        
        bits_per_batch = ceil(self.m / self.iterations)

        s_0 = 0
        C_i = defaultdict(int)      


        for i in range(self.iterations):

            s_i = min(s_0 + bits_per_batch, self.m)
            s_0 = s_i

            D_i = {}
            hash = lambda x: (x & int(bits_per_batch*i/2))
            for val in C_i.keys():
                D_i[hash(val)] = D_i.get(hash(val), []) + [val]

            # print("Privacy mechanism type:", self.privacy_mechanism_type)
            privacy_module = PrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type, bits_per_batch=bits_per_batch, batch=i)
            # mechanism = privacy_mechanism(
            #     self.varepsilon, D_i, self.privacy_mechanism_type)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []
            
            
        # ---------------------------clients --------------------------------

            for client in self.clients:
                prefix_client = client >> (self.m-s_i)
                response = mechanism(prefix_client)
                # clients_responses.append(response)
                p = random() 
                if p >= self.connection_loss_rate:
                    clients_responses.append(response)

        # ---------------------------clients --------------------------------

        # ---------------------------server --------------------------------
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
    n = 100000

    m = 8
    k = 9
    init_varepsilon = 0.2
    step_varepsilon = 0.3
    max_varepsilon = 3
    iterations = 4

    round = 5

    privacy_mechanism_type = "PreHashing" # ["GRR", "None","OUE", "PreHashing"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]

    server = FAServerPreHashing(n, m, k, init_varepsilon, iterations, round, \
        privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type)
    server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

    visualize_frequency(server.clients, server.C_truth, distribution_type=server.client_distribution_type)
