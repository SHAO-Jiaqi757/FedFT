"""_summary_
This experiment's configuration is defined fixed bits length that clients send to server every batch.
client size is increasing for each batch.
Returns:
    _type_: _description_
"""
from math import ceil
import pickle
import random
from typing import Dict, List
from Tree import TrieNumeric
from privacy_module import PrivacyModule
from server import FAServerPEM


# random.seed(0)

class FAServer(FAServerPEM):
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, 
        clients: List = [], C_truth: List = [], privacy_mechanism_type: List = "GRR_X", evaluate_type: str = "F1", 
        connection_loss_rate: float = 0, is_uniform_size: bool=False):
            super().__init__(n, m, k, varepsilon, iterations, round, clients, C_truth, privacy_mechanism_type, evaluate_type, connection_loss_rate, is_uniform_size)
            self.bits_per_iter =ceil(self.m / self.iterations) 
            self.trie = TrieNumeric(self.bits_per_iter, k = k)

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
        A_i = {"": 0}

        participants = 0
        for i in range(self.iterations):
            s_i = min(s_0 + self.bits_per_iter, self.m) # total bit length in this iteration
            delta_s = s_i - s_0 # required bit length from each client

            privacy_module = PrivacyModule(self.varepsilon, A_i, type=self.privacy_mechanism_type, required_bits = delta_s, s_i = s_0)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []

            if self.is_uniform_size : adder = int(self.n/self.iterations)
            
            else: adder = int(self.n/(2* self.iterations) + i * self.n / (self.iterations* (self.iterations + 1)))
            # print(f"Sampling {adder} clients")
            end_participants = participants + adder

            if i == self.iterations -1:
                end_participants = self.n

            for client in self.clients[participants: end_participants]:
                prefix_client = client[:s_i]
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
                if v and count > 0:
                    # A_i[v] = count
                    A_i[v] = 0  # validate v in next iteration
                    self.trie.insert((v), count) # count is stored in trie
                 
            s_0 = s_i
            # print(f"Group {i} generated: {A_i}")
        return A_i


if __name__ == '__main__':
    n = 3000
    m = 16
    k = 5
    init_varepsilon = 12
    step_varepsilon = 0.4
    max_varepsilon = 9
    iterations = 8

    round = 20

    privacy_mechanism_type = "GRRX" # ["GRR", "None","OUE"]
    evaluate_module_type = "F1" # ["NDCG", "F1"]
    
    with open("client_pois_500_bit.txt", 'rb') as f:
        clients = pickle.load(f)
    with open("top_5_client_pois_500_bit.txt", 'rb') as f:
        truth_top_k = pickle.load(f)
    truth_top_k = [int(str(i), 2) for i in truth_top_k]
    # n = 99413
    # save_path_dir = f""  # result path 
    # truth_top_k, clients = load_clients(filename=f"./dataset/zipf_{n}.txt", k=k)  # load clients from .txt

    server = FAServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k,\
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, 
        )
    
    # server.server_run_plot_varepsilon(init_varepsilon, step_varepsilon,max_varepsilon)
    server.server_run()

    # server.trie.item_start_with("1000101010001")
   