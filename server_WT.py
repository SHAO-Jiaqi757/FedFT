"""_summary_
This experiment's configuration is defined fixed bits length that clients send to server every batch.
client size is increasing for each batch.
Returns:
    _type_: _description_
"""
from math import ceil, log
import random
from symbol import parameters
from typing import Dict, List

from privacy_module import PrivacyModule
from server import FAServerPEM

from utils import plot_all_in_one, visualize_frequency, weight_score


# random.seed(0)

class WTServer(FAServerPEM):

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
        C_i = {}
        C_i[0] = 0 # initial weight_score
        adder_base = int((2*self.n)/((self.iterations*(self.iterations+1)))) 
        participants = 0
        bits_per_batch = ceil(self.m / self.iterations)

        for i in range(self.iterations):
            s_i = min(s_0 + bits_per_batch, self.m)
            delta_s = s_i - s_0
            s_0 = s_i

            D_i = {}
            for val in C_i.keys():
                for offset in range(2**delta_s):
                    D_i[(val << delta_s) + offset] = C_i[val]/(2**delta_s) if self.WT else 0 # inherit weight_score
                    # D_i[(val << delta_s) + offset] = 0


            privacy_module = PrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type, batch=i+1, WT=self.WT, s_i = s_i)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []

            if self.is_uniform_size : adder = int(self.n/self.iterations)
            
            else: adder = (i+1)*adder_base
            print(f"Sampling {adder} clients")
            end_participants = participants + adder

            if i == self.iterations -1:
                end_participants = self.n-1

            for client in self.clients[participants: end_participants+1]:
                prefix_client = client >> (self.m-s_i)
                response = mechanism(prefix_client)
           
                p = random.random() 
                if p >= self.connection_loss_rate:
                    clients_responses.append(response)
            participants = end_participants

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

    m = 16
    k = 9
    init_varepsilon = 0.2
    step_varepsilon = 0.4
    max_varepsilon = 3
    iterations = 9

    round = 20

    privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]
    
    # ----Weight Tree---- # 
    server = WTServer(n, m, k, init_varepsilon, iterations, round, \
        privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type, \
            WT =True
        )

    # server.server_run()
    xn, yn = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)
