from math import ceil, log
from random import random
from typing import Dict, List
from privacy_module import PEMPrivacyModule
from evaluate_module import EvaluateModule
from utils import plot_single_line, sort_by_frequency
import numpy as np

np.random.seed(123499)

class FAServerPEM():
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, clients: List = [], C_truth: List = [], privacy_mechanism_type: List = "GRR", evaluate_type: str = "F1", connection_loss_rate: float = 0):
        """_summary_

        Args:
            n (int): client size
            m (int): binary-string length
            k (int): top-k heavy hitters
            varepsilon (float): privacy budget
            iterations (int): number of groups
            round (int): running rounds
            clients (list): clients' items, one client has one data, default = []
            C_truth (list): truth ordered top-k items, default = []
            privacy_mechanism_type (str): local differential privacy mechanism. default is GRR (options: GRR, OUE, GRR_Weight, None)
            evaluate_type (str): evaluate function to estimate performance (NDCG or F1)
        """
        self.n = n
        self.connection_loss_rate = connection_loss_rate
        self.m = m
        self.k = k
        self.varepsilon = varepsilon
        self.iterations = iterations
        self.round = round
        self.clients = clients
        self.evaluate_type = evaluate_type
        self.evaluate_module = EvaluateModule(self.k, self.evaluate_type)

        self.__available_data_distribution = ["poisson", "uniform"]
        self.__available_privacy_mechanism_type = ["GRR", "None", "OUE", "PreHashing","GRR_Weight"]

        self.__init_privacy_mechanism(privacy_mechanism_type)

        if not self.clients:
            self.__init_clients()
        else:
            self.n = len(self.clients)
        if not C_truth:
            self.C_truth = sort_by_frequency(self.clients, self.k)
        else: self.C_truth = C_truth

    def __init_privacy_mechanism(self, privacy_mechanism_type: str):
        self.privacy_mechanism_type = privacy_mechanism_type if privacy_mechanism_type in self.__available_privacy_mechanism_type else "GRR"
        print(f"Privacy Mechanism: {self.privacy_mechanism_type}")

    def __init_clients(self):
        type = input(
            f"simulate client data with ___ distribution {self.__available_data_distribution}: ")
        if type not in self.__available_data_distribution:
            print("Invalid distribution type:: Default distribution will be 'poisson'")
            type = "poisson"
        self.client_distribution_type = type
        self.__simulate_client(type)()

    def __simulate_client_poisson(self, mu=None, var=None):
        if mu is None and var is None:

            mu = float(input("mean:"))

        clients = np.random.poisson(mu, self.n)
        self.clients = clients

    def __simulate_client_uniform(self, low=None, high=None):
        if low is None and high is None:

            low = int(input("low:"))
            high = int(input("high:"))

        clients = np.random.randint(low, high, self.n)
        clients = np.absolute(clients.astype(int))
        self.clients = clients

    def __simulate_client(self, type: str):
        if type == "poisson":
            return self.__simulate_client_poisson
        elif type == "uniform":
            return self.__simulate_client_uniform
        else:
            raise ValueError(
                f"Invalid client distribution type! Available types: {self.__available_data_distribution}")

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
        s_0 = ceil(log(self.k, 2))
        C_i = {}
        for i in range(2**s_0):
            C_i[i] = 0
        group_size = self.n//self.iterations
        for i in range(1, self.iterations+1):
            s_i = s_0 + ceil(i*(self.m-s_0)/self.iterations)
            delta_s = ceil(i*(self.m-s_0)/self.iterations) - \
                ceil((i-1)*(self.m-s_0)/self.iterations)
            D_i = {}
            for val in C_i.keys():
                for offset in range(2**delta_s):
                    D_i[(val << delta_s) + offset] = 0

            privacy_module = PEMPrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type)
            # mechanism = privacy_mechanism(
            #     self.varepsilon, D_i, self.privacy_mechanism_type)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []
            
            for client in self.clients[(i-1)*group_size: (i)*group_size]:
                prefix_client = client >> (self.m-s_i)
                response = mechanism(prefix_client)
                
                p = random() 
                if p >= self.connection_loss_rate:
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

    def server_run(self):
        evaluate_score = 0
        for rnd in range(self.round):
            np.random.shuffle(self.clients)
            
            C_i = self.predict_heavy_hitters()

            C_i = list(C_i.keys())
            print(f"Truth ordering: {self.C_truth}")
            print(f"Predicted ordering: {C_i}")

            evaluate_score += self.evaluate_module.evaluate(self.C_truth, C_i)
        evaluate_score /= self.round
        print(
            f"ROUND {rnd} :: varepsilon = {self.varepsilon}, {self.evaluate_type}= {evaluate_score:.2f}")
        return self.varepsilon, evaluate_score

    def server_run_plot_varepsilon(self, min_varepsilon, step_varepsilon, max_varepsilon):
        self.varepsilon = min_varepsilon
        varepsilon_list = []
        evaluate_score_list = []
        while self.varepsilon < max_varepsilon:
            varepsilon, evaluate_score = self.server_run()
            varepsilon_list.append(varepsilon)
            evaluate_score_list.append(evaluate_score)
            self.varepsilon += step_varepsilon

        plot_single_line(varepsilon_list, evaluate_score_list, "varepsilon",
                         f"{self.evaluate_type}", f"{self.evaluate_type} vs varepsilon", k=self.k)
        return varepsilon_list, evaluate_score_list


if __name__ == '__main__':
    n = 10000

    m = 4*10
    k = 8
    init_varepsilon = 0.2
    step_varepsilon = 0.8
    max_varepsilon = 12
    iterations =10

    round = 10

    evaluate_module_type = "F1" # ["NDCG", "F1"]

    privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]

    server = FAServerPEM(n, m, k, init_varepsilon, iterations, round,
         privacy_mechanism_type = privacy_mechanism_type, evaluate_type=evaluate_module_type, \
        )
    server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

    # visualize_frequency(server.clients, server.C_truth, server.client_distribution_type)
    