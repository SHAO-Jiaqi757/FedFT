from math import ceil, log
from random import random, uniform
from typing import Dict, List
from evaluate_module import EvaluateModule
from privacy_module.privacy_module import PrivacyModule
from utils import load_clients, plot_single_line, sort_by_frequency
import numpy as np

np.random.seed(0)

class FAServerPEM():
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, 
    clients: List = [], C_truth: List = [], privacy_mechanism_type: List = "GRR", evaluate_type: str = "F1", 
    connection_loss_rate: float = 0, WT: bool =False, is_uniform_size: bool=True):
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
            privacy_mechanism_type (str): local differential privacy mechanism. default is GRR (options: GRR, OUE, GRR_X, None)
            evaluate_type (str): evaluate function to estimate performance (NDCG or F1)
            connection_loss_rate (float)
            WT (boolean): is using Weight Trie?
            is_uniform_size: same size per iteration?

        """
        self.n = n
        self.connection_loss_rate = connection_loss_rate
        self.m = m
        self.k = k
        self.varepsilon = varepsilon
        self.iterations = iterations
        self.round = round
        self.clients = clients
        self.WT = WT
        self.is_uniform_size  = is_uniform_size

        self.evaluate_type = evaluate_type
        self.evaluate_module = EvaluateModule(self.k, self.evaluate_type)

        self.__available_data_distribution = ["poisson", "uniform"]
        self.__available_privacy_mechanism_type = ["GRR", "None", "OUE","GRR_X"]

        self.__init_privacy_mechanism(privacy_mechanism_type)

        if not len(self.clients):
            self.__init_clients()
        else:
            self.n = len(self.clients)
        if not len(C_truth):
            self.C_truth = sort_by_frequency(self.clients, self.k)
        else: self.C_truth = C_truth

    def __init_privacy_mechanism(self, privacy_mechanism_type: str):
        self.privacy_mechanism_type = privacy_mechanism_type if privacy_mechanism_type in self.__available_privacy_mechanism_type else "GRR"
        print(f"Privacy Mechanism: {self.privacy_mechanism_type}")

    def __init_clients(self):
        type = input(
            f"simulate client data with ___ distribution {self.__available_data_distribution}: ")
        type = type.strip()
        if type not in self.__available_data_distribution:
            print("Invalid distribution type:: Default distribution will be 'poisson'")
            type = "poisson"
        self.client_distribution_type = type
        self.__simulate_client(type)()

    def __simulate_client_poisson(self, mu=None, var=None):
        if mu is None and var is None:

            mu = float(input("mean:"))
        print(f"Generate {self.n} clients with [Poisson (mu={mu}]")
        clients = np.random.poisson(mu, self.n)
        self.clients = clients

    def __simulate_client_uniform(self, low=None, high=None):
        if low is None and high is None:

            low = int(input("low:"))
            high = int(input("high:"))

        print(f"Generate {self.n} clients with [Uniform ({low}, {high})]")
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
        adder_base = int((2*self.n)/((self.iterations*(self.iterations+1)))) 
        participants = 0
        for i in range(1, self.iterations+1):
            s_i = s_0 + ceil(i*(self.m-s_0)/self.iterations)
            delta_s = ceil(i*(self.m-s_0)/self.iterations) - \
                ceil((i-1)*(self.m-s_0)/self.iterations)

            # print("[PEM] bits/iter:", delta_s)
            D_i = {}
            for val in C_i.keys():
                for offset in range(2**delta_s):
                    D_i[(val << delta_s) + offset] = C_i[val]/(2**delta_s) if self.WT else 0 # inherit weight_score

            privacy_module = PrivacyModule(self.varepsilon, D_i, type=self.privacy_mechanism_type, batch=i, WT=self.WT, s_i = s_i)
            mechanism = privacy_module.privacy_mechanism()
            handle_response = privacy_module.handle_response() 
            clients_responses = []

            if self.is_uniform_size : adder = int(self.n/self.iterations)
            
            else: adder = (i)*adder_base
            print(f"Sampling {adder} clients")
            end_participants = participants + adder

            if i == self.iterations:
                end_participants = self.n-1

            for client in self.clients[participants: end_participants+1]:
                prefix_client = client >> (self.m-s_i)
                response = mechanism(prefix_client)
                p = random() 
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

