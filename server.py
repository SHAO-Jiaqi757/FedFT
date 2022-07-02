from math import ceil, log
from typing import Dict, List
from privacy_module import PrivacyModule 
from utils import NDCG, plot_single_line, sort_by_frequency, visualize_frequency
import numpy as np

np.random.seed(123499)

class FAServer():
    def __init__(self, n: int, m: int, k: int, varepsilon: float, g: int, clients: List = [], privacy_mechanism_type: List = "GRR"):
        """_summary_

        Args:
            n (int): client size
            m (int): binary-string length
            k (int): top-k heavy hitters
            varepsilon (float): privacy budget
            g (int): number of groups
        """
        self.n = n
        self.m = m
        self.k = k
        self.varepsilon = varepsilon
        self.g = g
        self.clients = clients

        self.__available_data_distribution = ["gaussian", "uniform"]
        self.__available_privacy_mechanism_type = ["GRR", "None"]

        self.__init_privacy_mechanism(privacy_mechanism_type)

        if not self.clients:
            self.__init_clients()
        self.C_truth = sort_by_frequency(self.clients, self.k)


    def __init_privacy_mechanism(self, privacy_mechanism_type: str):
        self.privacy_mechanism_type = privacy_mechanism_type if privacy_mechanism_type in self.__available_privacy_mechanism_type else "GRR"
        print(f"Privacy Mechanism: {self.privacy_mechanism_type}")

    def __init_clients(self):
        type = input(
            f"simulate client data with ___ distribution {self.__available_data_distribution}: ")
        if type not in self.__available_data_distribution:
            print("Invalid distribution type:: Default distribution will be 'gaussian'")
            type = "gaussian"
        self.__simulate_client(type)()

    def __simulate_client_gaussian(self, mu=None, var=None):
        if mu is None and var is None:

            mu = float(input("mean:"))
            var = float(input("variance:"))

        clients = np.random.normal(mu, var, self.n)
        clients = np.absolute(clients.astype(int))
        self.clients = clients

    def __simulate_client_uniform(self, low=None, high=None):
        if low is None and high is None:

            low = int(input("low:"))
            high = int(input("high:"))

        clients = np.random.randint(low, high, self.n)
        clients = np.absolute(clients.astype(int))
        self.clients = clients

    def __simulate_client(self, type: str):
        if type == "gaussian":
            return self.__simulate_client_gaussian
        elif type == "uniform":
            return self.__simulate_client_uniform
        else:
            raise ValueError(
                "Invalid client distribution type! Available types: [gaussian, uniform]")

    def predict_heavy_hitters(self) -> Dict:
        """_summary_

        Args:
            n (int): client size
            m (int): binary-string length
            k (int): top-k heavy hitters
            varepsilon (float): privacy budget
            g (int): number of groups
        Returns:
            Dict: top-k heavy hitters C_g and their frequencies.
        """
        s_0 = ceil(log(self.k, 2))
        C_i = {}
        for i in range(2**s_0):
            C_i[i] = 0
        group_size = self.n//self.g
        for i in range(1, self.g+1):
            s_i = s_0 + ceil(i*(self.m-s_0)/self.g)
            delta_s = ceil(i*(self.m-s_0)/self.g) - \
                ceil((i-1)*(self.m-s_0)/self.g)
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
                if count > 0:
                    C_i[v] = count
            # print(f"Group {i} generated: {C_i}")
        return C_i

    def server_run(self, round):

        ndcg = 0
        for rnd in range(round):
            np.random.shuffle(self.clients)

            C_i = self.predict_heavy_hitters()

            C_i = list(C_i.keys())
            print(f"Truth ordering: {self.C_truth}")
            print(f"Predicted ordering: {C_i}")

            ndcg += NDCG(self.C_truth, C_i, self.k)
        ndcg /= round
        print(
            f"ROUND {rnd} :: varepsilon = {self.varepsilon}, NDCG = {ndcg:.2f}")
        return self.varepsilon, ndcg

    def server_run_plot_varepsilon(self, round, min_varepsilon, step_varepsilon, max_varepsilon):
        self.varepsilon = min_varepsilon
        varepsilon_list = []
        ndcg_list = []
        while self.varepsilon < max_varepsilon:
            varepsilon, ndcg = self.server_run(round)
            varepsilon_list.append(varepsilon)
            ndcg_list.append(ndcg)
            self.varepsilon += step_varepsilon

        plot_single_line(varepsilon_list, ndcg_list, "varepsilon",
                         "NDCG", "NDCG vs varepsilon", k=self.k)


if __name__ == '__main__':
    n = 1000

    m = 16
    k = 9
    init_varepsilon = 0.2
    step_varepsilon = 0.5
    max_varepsilon = 12
    g = 9

    round = 50

    privacy_mechanism_type = "GRR"

    server = FAServer(n, m, k, init_varepsilon, g, privacy_mechanism_type = privacy_mechanism_type)
    server.server_run_plot_varepsilon(
        round, init_varepsilon,  step_varepsilon, max_varepsilon)

    visualize_frequency(server.clients, server.C_truth)