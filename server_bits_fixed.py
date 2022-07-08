from math import ceil, log
from typing import Dict, List

from privacy_module import PrivacyModule
from server import FAServerPEM

from utils import visualize_frequency


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
        
        group_size = self.n//self.batch_size

        bits_per_batch = ceil(self.m / self.batch_size)
        s_0 = bits_per_batch
        C_i = {}
        for i in range(2**s_0):
            C_i[i] = 0

        for i in range(self.batch_size):
   
            s_i = min(s_0 + bits_per_batch, self.m)
            delta_s = s_i - s_0
            s_0 = s_i

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

            for client in self.clients[(i)*group_size: (i+1)*group_size]:
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

    m = 32
    k = 9
    init_varepsilon = 0.2
    step_varepsilon = 0.5
    max_varepsilon = 12 
    batch_size = 9

    sampling_rate = 1
    round = 50

    privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]
    evaluate_module_type = "NDCG" # ["NDCG", "F1"]

    server = FAServerPEM(n, m, k, init_varepsilon, batch_size, round, privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type, \
        sampling_rate= sampling_rate)
    server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

    visualize_frequency(server.clients, server.C_truth)
