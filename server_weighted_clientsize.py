"""_summary_
This experiment's configuration is defined fixed bits length that clients send to server every batch.
client size is increasing for each batch.
Returns:
    _type_: _description_
"""
from math import ceil, log
import random
from typing import Dict, List
from utils import load_clients
from privacy_module import PrivacyModule
from server import FAServerPEM



random.seed(0)

class ServerWeightClientSize(FAServerPEM):

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
        C_i[0] = 0 # initial weight_score
        

        for i in range(self.batch_size):

            s_i = min(s_0 + bits_per_batch, self.m)
            delta_s = s_i - s_0
            s_0 = s_i

            D_i = {}
            for val in C_i.keys():
                for offset in range(2**delta_s):
                    D_i[(val << delta_s) + offset] = C_i[val]/(2**delta_s) if self.privacy_mechanism_type=="GRR_Weight" else 0 # inherit weight_score

            # self.varepsilon = varepsilons[i]
            # print("Privacy mechanism type:", self.privacy_mechanism_type)
            privacy_module = PrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type, batch=i+1, bits_per_batch=bits_per_batch)
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
    n = 10000
    
    m = 4*10
    k = 8
    init_varepsilon = 0.2
    step_varepsilon = 0.8
    max_varepsilon = 12
    batch_size =10

    sampling_rate = 1
    round = 10

    truth_top_k, clients = load_clients(filename="./dataset/triehh_clients_remove_top5_9004.txt", k=k)
    # truth_top_k =[], clients = []
    privacy_mechanism_type = "GRR_Weight" # ["GRR", "None","OUE"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]

    # ----Weight Tree & Client Size fitting---- # 
    server = ServerWeightClientSize(n, m, k, init_varepsilon, batch_size, round, clients=clients, C_truth = truth_top_k, \
        privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type, \
        sampling_rate= sampling_rate)

    xc, yc = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)
    # server.server_run()


