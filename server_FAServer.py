"""
FAServer will be updated to that in 'server_FAServer_bit.py' in the future.
"""
from math import ceil, log
import random
from typing import Dict, List

from privacy_module import PrivacyModule
from server import FAServerPEM

from utils import load_clients


# random.seed(0)

class FAServer(FAServerPEM):
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, 
        clients: List = [], C_truth: List = [], privacy_mechanism_type: List = "GRR_X", evaluate_type: str = "F1", 
        connection_loss_rate: float = 0, is_uniform_size: bool=False):
            super().__init__(n, m, k, varepsilon, iterations, round, clients, C_truth, privacy_mechanism_type, evaluate_type, connection_loss_rate)
            self.bits_per_iter =ceil(self.m / self.iterations) 
            self.trie = TrieNumeric(self.bits_per_iter, k = k)
            self.is_uniform_size = is_uniform_size
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

        for i in range(self.iterations):
            s_i = min(s_0 + bits_per_batch, self.m)
            delta_s = s_i - s_0
            s_0 = s_i

            privacy_module = PrivacyModule(self.varepsilon, A_i, type=self.privacy_mechanism_type,s_i = s_i, required_bits = delta_s)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []

            if self.is_uniform_size : adder = int(self.n/self.iterations)
            
            else: adder = int((self.n / (2*self.iterations)) + (i) * self.n / (self.iterations* (self.iterations + 1)))
            # else: adder = (i+1)*int((2*self.n)/((self.iterations*(self.iterations+1)))) 
            print(f"Sampling {adder} clients")
            end_participants =  + participants + adder

            if i == self.iterations -1:
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
            for indx in range(min(self.k, len(C_i_sorted))):
                v, count = C_i_sorted[indx]
                if count > 0:
                    A_i[v] = 0  # validate v in next iteration
                    # self.trie.insert((v), count) # count is stored in trie
                 
            # print(f"Group {i} generated: {A_i}")
        return A_i


if __name__ == '__main__':
    m = 64
    k = 5
    init_varepsilon = 0.5
    step_varepsilon = 0.6
    max_varepsilon =  9
    iterations = 32

    round = 20

    privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]
    n = 99413
    save_path_dir = f""  # result path 
    truth_top_k, clients = load_clients(filename=f"./dataset/zipf_{n}.txt", k=k)  # load clients from .txt

    server = FAServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, 
        )
    
    x_xtf, y_xtf = server.server_run_plot_varepsilon(
    init_varepsilon,  step_varepsilon, max_varepsilon)
    server.server_run()
    # print(([int(bit_string, 2) for bit_string in server.trie.display_trie(is_get_hhs = True)]))

    # server.trie.item_start_with("1000101010001")
   