"""
PEM
"""
from math import ceil, log
import random
from typing import Dict, List
from evaluate_module import EvaluateModule
from privacy_module.privacy_module import PrivacyModule
from utils import load_clients, plot_single_line, sort_by_frequency
import numpy as np

np.random.seed(0)
random.seed(0)
class BaseServer():
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, 
    clients: List = [], C_truth: List = [], privacy_mechanism_type: List = "GRR", evaluate_type: str = "F1", 
    connection_loss_rate: float = 0):
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
        self.evaluate_module = EvaluateModule( self.evaluate_type)

        self.__available_data_distribution = ["poisson", "uniform", "normal"]
        self.__available_privacy_mechanism_type = ["GRR", "None", "GRR_X", "OHE_2RR", "PIRAPPOR"]

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

    def __simulate_client_poisson(self, mu=None):
        if mu is None:

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

    def __simulate_client_normal(self, mu=None, var=None):
        if mu is None and var is None:

            mu = float(input("mean:"))
            sigma = float(input("standard deviation:"))
        print(f"Generate {self.n} clients with [Normal (mu={mu}, sigma={sigma})]")
        gen_data = np.random.normal(mu, sigma, self.n).astype(int)
        gen_data_filter = gen_data > 0 
        clients =gen_data[gen_data_filter]
        self.clients = clients

    def __simulate_client(self, type: str):
        if type == "poisson":
            return self.__simulate_client_poisson
        elif type == "uniform":
            return self.__simulate_client_uniform
        elif type == "normal":
            return self.__simulate_client_normal
        else:
            raise ValueError(
                f"Invalid client distribution type! Available types: {self.__available_data_distribution}")

    def predict_heavy_hitters(self, stop_iter=-1) -> Dict:
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
        if stop_iter == -1: stop_iter = self.iterations # used to accurate in bits (mode query)

        s_0 = ceil(log(self.k, 2))
        A_i = {}
        for i in range(2**s_0):
            A_i[i] = 0
        participants = 0

        for i in range(1, self.iterations+1):
            if stop_iter < i:
                print(self.m-s_i+1)
                return dict((key<<(self.m-s_0), value) for (key, value) in A_i.items())

            s_i = s_0 + ceil(i*(self.m-s_0)/self.iterations)
            delta_s = ceil(i*(self.m-s_0)/self.iterations) - \
                ceil((i-1)*(self.m-s_0)/self.iterations)


            clients_responses = []

            adder = int(self.n/self.iterations)

            print(f"Sampling {adder} clients")
            end_participants = participants + adder

    
            if i== stop_iter:
                end_participants = self.n
            for client_id in range(participants, end_participants):
                client_v = self.clients[client_id]
                privacy_module = PrivacyModule(self.varepsilon, A_i, type=self.privacy_mechanism_type, s_i = s_i, required_bits = delta_s, client_id=client_id)
                mechanism = privacy_module.privacy_mechanism()
                handle_response = privacy_module.handle_response() 
                prefix_client = client_v >> (self.m-s_i) # prefix s_i bits of the prefix value.
                response = mechanism(prefix_client)
                p = random.random() 
                if p >= self.connection_loss_rate and response is not None:
                    clients_responses.append(response)
            participants = end_participants 

            C_i = handle_response(clients_responses)

            C_i_sorted = sorted(C_i.items(), key=lambda x: x[-1], reverse=True)


            A_i = {}
            for indx in range(min(self.k, len(C_i_sorted))):
                v, count = C_i_sorted[indx]
                if count > 0:
                    A_i[v] = 0
            # print(f"Group {i} generated: {A_i}")
        return A_i

    def server_run(self):
        evaluate_score = 0
        for rnd in range(self.round):
            np.random.shuffle(self.clients)
            self.rnd = rnd 
            # self.varepsilon -= 0.01 * self.rnd
            print("eps", self.varepsilon)
            self.A_i = self.predict_heavy_hitters()
            
            estimate_top_k = list(self.A_i.keys())[:min(self.k, len(self.A_i))]
            
            print(f"[eps = {self.varepsilon}] Truth ordering: {self.C_truth}")
            print(f"[eps = {self.varepsilon}] Predicted ordering: {estimate_top_k}")
            score = self.evaluate_module.evaluate(self.C_truth, estimate_top_k)
            print(f"{self.evaluate_type}= {score}")
            evaluate_score += score 
        evaluate_score /= self.round
        print(
            f"[Server End]:: varepsilon = {self.varepsilon}, {self.evaluate_type}= {evaluate_score:.2f}")
        return self.varepsilon, evaluate_score

    def server_run_plot_varepsilon(self, min_varepsilon, step_varepsilon, max_varepsilon, is_plot = False):
        self.varepsilon = min_varepsilon
        varepsilon_list = []
        evaluate_score_list = []
        while self.varepsilon < max_varepsilon:
            varepsilon, evaluate_score = self.server_run()
            varepsilon_list.append(varepsilon)
            evaluate_score_list.append(evaluate_score)
            self.varepsilon += step_varepsilon
        if is_plot:
            plot_single_line(varepsilon_list, evaluate_score_list, "varepsilon",
                         f"{self.evaluate_type}", f"{self.evaluate_type} vs varepsilon", k=self.k)
        return varepsilon_list, evaluate_score_list

