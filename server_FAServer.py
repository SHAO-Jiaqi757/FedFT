"""
FA_Server will be updated to that in 'server_FAServer_bit.py' in the future.
"""
from math import ceil, log
import random, sys, json
from typing import Dict, List
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
import contextlib
from privacy_module import PrivacyModule
from evaluate_module import EvaluateModule
from server import BaseServer
from utils import distance, load_clients, blockPrint, enablePrint

random.seed(0)
# blockPrint()

class FAserver(BaseServer):
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, 
        clients: List = [], C_truth: List = [], privacy_mechanism_type: List = "GRR_X", evaluate_type: str = "F1", 
        connection_loss_rate: float = 0, is_uniform_size: bool=False, stop_iter = -1):
            super().__init__(n, m, k, varepsilon, iterations, round, clients, C_truth, privacy_mechanism_type, evaluate_type, connection_loss_rate)
            self.bits_per_iter =ceil(self.m / self.iterations) 
            # self.trie = TrieNumeric(self.bits_per_iter, k = k)
            self.is_uniform_size = is_uniform_size
            if stop_iter == -1:
                self.stop_iter = iterations
            else: 
                self.stop_iter = stop_iter
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

        for i in range(self.stop_iter):
            s_i = min(s_0 + bits_per_batch, self.m) # current bit length
            delta_s = s_i - s_0 # required bit length, e.g. 2bits per iter.
            s_0 = s_i # last iter's bit length


            clients_responses = []

            if self.is_uniform_size : adder = int(self.n/self.stop_iter)
            
            else: adder = int(self.n /(2*self.stop_iter) + (i) * self.n / (self.stop_iter* (self.stop_iter + 1)))
            print(f"Sampling {adder} clients")
            end_participants = participants + adder

            if i == self.stop_iter-1:
                end_participants = self.n-1
            for client_id in range(participants, end_participants+1):
                client_v = self.clients[client_id]
                privacy_module = PrivacyModule(self.varepsilon, A_i, type=self.privacy_mechanism_type,s_i = s_i, required_bits = delta_s, client_id=client_id)
                mechanism = privacy_module.privacy_mechanism()
                
                if self.privacy_mechanism_type=="GRR_X" and privacy_module.p <= 0.5:
                    privacy_module = PrivacyModule(self.varepsilon, A_i, type="GRR",s_i = s_i, required_bits = delta_s, client_id=client_id)
                    mechanism = privacy_module.privacy_mechanism() 
                    
                handle_response = privacy_module.handle_response() 
                
                prefix_client = client_v >> (self.m-s_i)
                response = mechanism(prefix_client)
                # simulate connection loss rate
                p = random.random() 
                if p >= self.connection_loss_rate and response is not None:
                    clients_responses.append(response)
            participants = end_participants

            C_i = handle_response(clients_responses)

            C_i_sorted = sorted(C_i.items(), key=lambda x: x[-1], reverse=True)


            A_i = {}
            # a  = self.k*2**self.bits_per_iter if i==self.stop_iter else self.k
            a = len(C_i_sorted) if i==self.stop_iter-1 else self.k
            for indx in range(min(a, len(C_i_sorted))):
                v, count = C_i_sorted[indx]
                if count > 0:
                    A_i[v] = 0  # validate v in next iteration
                    if i == self.stop_iter-1: A_i[v] = count
            if not A_i:
                A_i = {0: 0}
        self.accurate_bits = self.m-s_0
        if self.accurate_bits:
            print(f"accurate bits: {self.accurate_bits}")
            A_tmp = A_i
            A_i = {}
            for i in A_tmp.keys():
                # change key  A_i[i << self.accurate_bits] = A_i[i]
                A_i[i << self.accurate_bits] = A_tmp[i]

        return A_i


def FA_cluster(clients: list, k: int, evaluate_module_type="recall", m=64, iterations=32, varepsilon=2, connection_loss_rate=0, stop_iter=-1):
    """_summary_

    Args:
        clients (list): clients' data
        k (int): top k
        evaluate_module_type (str, optional): performace evaluation "recall", "F1" . Defaults to "recall".
        m (int, optional): maximum length of binary strings. Defaults to 64.
        iterations (int, optional): number of iterations/trie's height. Defaults to 32.
        varepsilon (float, optional): LDP privacy parameter. Defaults to 2.
    return:
        str: json string of one cluster's result
    """
    # encode clients into bitstrings
    round = 1

    n = len(clients)
    # print("[debug]:: n", n)
    server = FAserver(n, m, k, varepsilon, iterations, round, clients=clients, C_truth=None,
                         evaluate_type=evaluate_module_type, connection_loss_rate=connection_loss_rate,
                         is_uniform_size=False, 
                            stop_iter=stop_iter
                         )

    x_xtf, _ = server.server_run()

    predict_hh = server.A_i

    results = {'predict_hh': predict_hh,  'eps': x_xtf}
    return json.dumps(results)



@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

 
 
def FA_aggregation(clients: list, k: int, global_truth_top_k: list, saved_csv_local_global, varepsilon: float = 2, evaluate_type="recall", m=64, iterations=32):
    """_summary_

    Args:
        n (int): number of clients
        clients (list[list]): list of clusters of clients
        k (int): top k
        global_truth_top_k (list): global truth top k
        var_epsilon (float, optional): LDP privacy parameter. Defaults to 2.
        evaluate_type (str, optional): performace evaluation "recall", "F1" . Defaults to "recall".
        cluster_size (int, optional): number of clients in one cluster . Defaults to 5000.
        m (int, optional): maximum length of binary strings. Defaults to 64.
        iterations (int, optional): number of iterations/trie's height. Defaults to 32.

    Returns:
        float: score (F1 or recall)
    """
    
    clusters = len(clients)
    total_clients = sum(len(clients[i]) for i in range(clusters))
    blockPrint()
    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        results = Parallel(n_jobs=clusters)(delayed(FA_cluster)(
            clients[i], k, evaluate_type, m, iterations, varepsilon) for i in tqdm(range(clusters)))
    enablePrint()
    candidates_among_clusters = {}
    tmp_csv_row = []
    for i in range(len(results)):
        result_i = json.loads(results[i])
        predict_hh = result_i['predict_hh']
        tmp_csv_row.append(f"{list(predict_hh.keys())[:k]}")
        for hh in predict_hh:
            x = predict_hh[hh] 
            if x <= k:
                continue
            incre = x/sum(predict_hh.values())
            candidates_among_clusters[hh] = candidates_among_clusters.get(hh, 0) + incre

        # aggregate results
    candidates_among_clusters = sorted(candidates_among_clusters.items(),
                           key=lambda x: x[1], reverse=True)

    candidates_n = len(candidates_among_clusters)
    
    
    good_hhs = []
    bad_hh = []
    for i in range(candidates_n):
        x = int(candidates_among_clusters[i][0])
        if x in bad_hh:
            continue        
        for j in range(i+1, candidates_n):

            y = int(candidates_among_clusters[j][0])
            threshold = (min(len(bin(x)), len(bin(y))) - 2)/2
            dis = distance(x, y)
            if dis <= threshold and candidates_n - len(bad_hh) > k:
                bad_hh.append(y)
                continue
            # print(f"{x} {y}: distance = {dis}")
        good_hhs.append(x)
        if len(good_hhs) == k: break
    print("global truth_top_k: ", global_truth_top_k)
    evaluate_module = EvaluateModule(evaluate_type)
    score = evaluate_module.evaluate(global_truth_top_k, good_hhs)
    print("finall hhs: ", good_hhs, " score:", score)

    print("Done!")
    tmp_csv_row.append(f"{good_hhs}")
    tmp_csv_row.append(f"{global_truth_top_k}")

    
    saved_csv_local_global.append(tmp_csv_row)
    return score, saved_csv_local_global
 
def FA_running_rounds(n: int, clients: list, k: int, global_truth_top_k: list, varepsilon: float = 2, step_varepsilon=0.4, max_varepsilon=2.2,
                         evaluate_type="recall", cluster_size=5000, m=64, iterations=32, rounds=10):
    """_summary_

    Args:
        n (int): number of clients
        clients (list): clients' data
        k (int): top k
        global_truth_top_k (list): global truth top k
        varepsilon (float, optional): LDP privacy parameter. Defaults to 2.
        step_varepsilon (float, optional): step of varepsilon. Defaults to 0.4.
        max_epsilon (float, optional): max varepsilon. Defaults to 2.2.
        cluster_size (int, optional): number of clients in one cluster . Defaults to 5000.
        m (int, optional): maximum length of binary strings. Defaults to 64.
        iterations (int, optional): number of iterations/trie's height. Defaults to 32.
        rounds (int, optional): number of rounds. Defaults to 10.

    Returns:
        list[float], list[float]: epsilons, scores
    """
    varepsilons = []
    scores = []
    while varepsilon < max_varepsilon:
        # running rounds
        score = 0
        for i in range(rounds):
            score += FA_aggregation(n, clients, k, global_truth_top_k, varepsilon, evaluate_type, cluster_size, m, iterations)
                  
        varepsilons.append(varepsilon)
        scores.append(score/rounds)
        varepsilon += step_varepsilon

    # average score
    return varepsilons, scores
