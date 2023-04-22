
import json
from joblib import Parallel, delayed
import os, sys, random
import numpy as np
from _utils import encode_file_initate, get_non_iid_clusters_topk
from utils import blockPrint, enablePrint, encode_words
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
 
from tqdm import tqdm
from evaluate_module import EvaluateModule
from exp_generate_words import load_words, load_words_count

from fedft import fedft_cluster, std_out_err_redirect_tqdm, distance
from intra_cluster_exp import DATA_PATH

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
        
    for i in range(len(results)):
        result_i = json.loads(results[i])
        predict_hh = result_i['predict_hh']
        
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
    return score

def load_clusters(k=5):
    
    clusters_ = []
    
    cluster_top_k = []
    for n in range(2000, 10001, 1500):
        filename = f"words_generate_{n}"
        cluster_truth_top_k = list(map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
        cluster = list(map(int,load_words(f"{DATA_PATH}{filename}_encode.txt")))
        
        clusters_.append(cluster)
        cluster_top_k.append(cluster_truth_top_k)
        
    truth_hh = get_non_iid_clusters_topk(cluster_top_k, k) 
    # truth_hh = get_bayes_shrinkage_topk(k)
    
    return clusters_, truth_hh

if __name__ == '__main__':

    m = 48
    k = 5
    # encode_file_initate(k)
    
    init_varepsilon = 0.5
    step_varepsilon = 1 
    max_varepsilon = 9.6
    iterations = 24

    runs = 40

    clients, truth_hh = load_clusters(k) 
    print("truth_hh: ", truth_hh)

    results = {}

    evaluate_module_type = "F1" # ["recall", "F1"]
    
    for varepsilon in np.arange(init_varepsilon, max_varepsilon, step_varepsilon):
         
        score_fedft = 0
        blockPrint() 
        for rnd in range(runs):
            score = fed_ft_aggregation(clients, k, global_truth_top_k=truth_hh, varepsilon=varepsilon,
                                    \
                evaluate_type = evaluate_module_type, m=m, iterations=iterations)
            score_fedft += score
        score = score_fedft/runs
        enablePrint()
        
        results[varepsilon] = score
        print(f"varepsilon: {varepsilon} score: {score}")
    
    # with open(f"inter_cluster_exp_{evaluate_module_type}_m_{m}_k_{k}_iter_{iterations}.json", "w") as f:
    #     json.dump(results, f)
    print(",".join([ str(round(x, 3)) for x in list(results.values())]))
    
