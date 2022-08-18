from math import ceil, exp, log
from typing import Dict, List

import numpy as np
from server import FAServerPEM
from privacy_module import PrivacyModule 
from evaluate_module import EvaluateModule
from utils import plot_single_line, visualize_frequency

class TrieHHPlus(FAServerPEM):
        def __init__(self, n: int, m: int, k: int, varepsilon: float, delta: float, sampling_rate: float, batch_size: int, round: int, clients: List = [], privacy_mechanism_type: List = "GRR", evaluate_type: str = "NDCG"):
            super().__init__(n, m, k, varepsilon, batch_size, round, clients, privacy_mechanism_type, evaluate_type, sampling_rate= sampling_rate)
            self.delta = delta 
            self.__init_theta()

        def __init_theta(self):
            alpha = self.sampling_rate/(1-exp(-self.varepsilon))
            self.theta  = log(self.delta)/(1/(1+alpha) - log(1/alpha))
            print(f"Theta: {self.theta}")
            

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
            s_0 = ceil(log(self.k, 2))
            C_i = {}
            for i in range(2**s_0):
                C_i[i] = 0
            group_size = self.n//self.batch_size
            for i in range(1, self.batch_size+1):
                s_i = s_0 + ceil(i*(self.m-s_0)/self.batch_size)
                delta_s = ceil(i*(self.m-s_0)/self.batch_size) - \
                    ceil((i-1)*(self.m-s_0)/self.batch_size)
                D_i = {}
                for val in C_i.keys():
                    for offset in range(2**delta_s):
                        D_i[(val << delta_s) + offset] = 0

                privacy_module = PrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type)
                # mechanism = privacy_mechanism(
                #     self.varepsilon, D_i, self.privacy_mechanism_type)
                mechanism = privacy_module.privacy_mechanism()
                handle_response = privacy_module.handle_response() 
                clients_responses = []
                for client in self.clients[(i-1)*group_size: i*group_size]:
                    prefix_client = client >> (self.m-s_i)
                    response = mechanism(prefix_client)
                    clients_responses.append(response)

                D_i = handle_response(clients_responses)

                D_i_sorted = sorted(D_i.items(), key=lambda x: x[-1], reverse=True)
                C_i = {}
                for indx in range(min(self.k, len(D_i_sorted))):
                    v, count = D_i_sorted[indx]
                    if count > self.theta:
                        C_i[v] = count
                # print(f"Group {i} generated: {C_i}")
            return C_i

        def server_run_plot_varepsilon(self, min_varepsilon, step_varepsilon, max_varepsilon):
            self.varepsilon = min_varepsilon
            varepsilon_list = []
            evaluate_score_list = []
            while self.varepsilon < max_varepsilon:
                varepsilon, evaluate_score = self.server_run()
                varepsilon_list.append(varepsilon)
                evaluate_score_list.append(evaluate_score)
                self.varepsilon += step_varepsilon
                self.__init_theta()

            plot_single_line(varepsilon_list, evaluate_score_list, "varepsilon",
                            f"{self.evaluate_type}", f"{self.evaluate_type} vs varepsilon", k=self.k)


if __name__ == '__main__':
    n = 1000

    m = 16
    k = 9 
    init_varepsilon = 0.2
    step_varepsilon = 0.1
    max_varepsilon = 1
    batch_size = 9

    round = 5
    sampling_rate = 0.5
    delta = 3
    privacy_mechanism_type = "GRR" # ["GRR", "None", "OUE"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]

    server = TrieHHPlus(n, m, k, init_varepsilon,
            delta = delta, 
            sampling_rate = sampling_rate, 
            batch_size=batch_size, round=round, 
            privacy_mechanism_type = privacy_mechanism_type,
            evaluate_type=evaluate_module_type)
    server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

    visualize_frequency(server.clients, server.C_truth, server.client_distribution_type)
