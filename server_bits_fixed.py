"""_summary_
This experiment's configuration is defined fixed bits length that clients send to server every batch.
client size is increasing for each batch.
Returns:
    _type_: _description_
"""
from math import ceil, log
import random
from typing import Dict, List

from privacy_module import PrivacyModule
from server import FAServerPEM

from utils import visualize_frequency


random.seed(0)

class FAServer(FAServerPEM):

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
        
        adder_base = ceil((2*self.n)/(self.batch_size*(self.batch_size+1)))

        bits_per_batch = ceil(self.m / self.batch_size)

        s_0 = 0
        C_i = {}
        C_i[0] = 0
        


        for i in range(self.batch_size):

            s_i = min(s_0 + bits_per_batch, self.m)
            delta_s = s_i - s_0
            s_0 = s_i

            D_i = {}
            for val in C_i.keys():
                for offset in range(2**delta_s):
                    D_i[(val << delta_s) + offset] = 0

            # print("Privacy mechanism type:", self.privacy_mechanism_type)
            privacy_module = PrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type)
            # mechanism = privacy_mechanism(
            #     self.varepsilon, D_i, self.privacy_mechanism_type)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []
            
            adder = (i+1)*adder_base
            
           
            print(f"Sampling {adder} clients")

            for client in random.choices(self.clients, k=adder):
                prefix_client = client >> (self.m-s_i)
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

    m = 56
    k = 9
    init_varepsilon = 0.2
    step_varepsilon = 0.3
    max_varepsilon = 3
    batch_size = 16

    sampling_rate = 1
    round = 20

    privacy_mechanism_type = "OUE" # ["GRR", "None","OUE"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]

    server = FAServer(n, m, k, init_varepsilon, batch_size, round, privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type, \
        sampling_rate= sampling_rate)
    server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

    visualize_frequency(server.clients, server.C_truth, distribution_type=server.client_distribution_type)