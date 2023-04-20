
from math import tanh
import json
import random
from joblib import Parallel, delayed
import os, sys, random
import pickle
import contextlib
import numpy as np
from tqdm.contrib import DummyTqdmFile
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
 
from server_AServer import Aserver
from tqdm import tqdm
from evaluate_module import EvaluateModule
from exp_generate_words import load_words

from fedft import fedft_cluster, std_out_err_redirect_tqdm, distance
from fedft_non_iid_exp.intra_cluster_exp import DATA_PATH

def fed_ft_aggregation(clients: list, k: int, global_truth_top_k: list, varepsilon: float = 2, evaluate_type="recall", m=64, iterations=32, connection_loss_rate=0):
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
    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        results = Parallel(n_jobs=clusters)(delayed(fedft_cluster)(
            clients[i], k, evaluate_type, m, iterations, varepsilon, connection_loss_rate) for i in tqdm(range(clusters)))

        candidates_among_clusters = {}
        
        for i in range(clusters):
            result_i = json.loads(results[i])
            print(result_i)
            
            predict_hh = result_i['predict_hh']
            
            for hh in predict_hh:
                x = predict_hh[hh] 
                incre = x * (1 - np.exp(-x)) * total_clients/len(clients[i])
                # incre = tanh(0.2*k + x) * total_clients/len(clients[i])

                candidates_among_clusters[hh] = candidates_among_clusters.get(hh, 0) + incre

        # aggregate results
        candidates_among_clusters = sorted(candidates_among_clusters.items(),
                           key=lambda x: x[1], reverse=True)[: k * 4]

        candidates_n = len(candidates_among_clusters)
        good_hhs = []
        bad_hh = []
        # threshold = 5
        for i in range(candidates_n):
            x = int(candidates_among_clusters[i][0])
            if x in bad_hh:
                continue
            threshold = (len(bin(x)) - 2)/2
            for j in range(candidates_n-1, i, -1):

                y = int(candidates_among_clusters[j][0])

                dis = distance(x, y)
                if dis <= threshold and candidates_n - len(bad_hh) > k:
                    bad_hh.append(y)
                    continue
                # print(f"{x} {y}: distance = {dis}")
            good_hhs.append(x)
            if len(good_hhs) == k: break
        # good_hhs = good_hhs[:k]
        print("global truth_top_k: ", global_truth_top_k)
        evaluate_module = EvaluateModule(evaluate_type)
        score = evaluate_module.evaluate(global_truth_top_k, good_hhs)
        print("finall hhs: ", good_hhs, " score:", score)

    print("Done!")
    return score




if __name__ == '__main__':

    m = 48
    k = 5
    
    init_varepsilon = 6.5
    step_varepsilon = 1 
    max_varepsilon = 9.6
    iterations = 24

    runs = 20

    clients = []
    cluster_top_k = []
    for n in range(2000, 10001, 1500):
        filename = f"words_generate_{n}"
        
        cluster_truth_top_k = list(map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
        cluster = list(map(int,load_words(f"{DATA_PATH}{filename}_encode.txt")))
        
        clients.append(cluster)
        cluster_top_k.append(cluster_truth_top_k)
        
    results = {}

    evaluate_module_type = "recall" # ["recall", "F1"]

    score_fedft = 0
    for rnd in range(runs):
        score = fed_ft_aggregation(clients, k, global_truth_top_k=[0], varepsilon=init_varepsilon,
                                   \
            evaluate_type = evaluate_module_type, m=m, iterations=iterations)
        score_fedft += score
    score = score_fedft/runs

    print(cluster_top_k)
    # if "FedFT" not in results:
    #     results["FedFT"] = {'y': [], 'x': []}
    # results["FedFT"]['x'].append(k)
