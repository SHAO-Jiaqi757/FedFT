# Inter-cluster experiment Real Data top 10 
import json
import pickle
from joblib import Parallel, delayed
import os, sys, random
import numpy as np
from _utils import get_non_iid_clusters_topk
from utils import blockPrint, enablePrint, encode_words, encode_file_initate
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
import pandas as pd

from tqdm import tqdm
from evaluate_module import EvaluateModule
from exp_generate_words import load_words, load_words_count

from fedft import fedft_cluster, std_out_err_redirect_tqdm, distance
from exp_intra_sentiment import DATA_PATH

save_path_dir = f"plots/exp_non_iid/"  # result path

saved_csv_local_global = []

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
    
    return score

def load_clusters(k=5):
    
    clusters_ = []
    word_count_file_list=[] 
    for cluster in ["sentiment", "reddit"]:
        filename = f"heavyhitters_{cluster}"
        encode_file_initate(k, filename= filename, datadir=DATA_PATH)

        # cluster_truth_top_k = list(map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
        cluster = list(map(int,load_words(f"{DATA_PATH}{filename}_encode.txt")))
        clusters_.append(cluster)
        word_count_file_list.append(f"{DATA_PATH}{filename}_count.txt")
        # cluster_top_k.append(cluster_truth_top_k)
        
    truth_hh = get_non_iid_clusters_topk(k, word_count_file_list)
    truth_hh = encode_words(truth_hh)
    # truth_hh = get_bayes_shrinkage_topk(k)
    
    return clusters_, truth_hh

if __name__ == '__main__':

    m = 48
    k = 10
    
    init_varepsilon = 0.5
    step_varepsilon = 1 
    max_varepsilon = 9.6
    iterations = 24
    
    runs = 20

    clients, truth_hh = load_clusters(k) 
    print("truth_hh: ", truth_hh)

    results = [] 

    for evaluate_module_type in ["recall", "F1"]:
        scores = []
        for varepsilon in np.arange(init_varepsilon, max_varepsilon, step_varepsilon):
            score_fedft = 0
            blockPrint() 
            for rnd in range(runs):
                score = fed_ft_aggregation(clients, k, global_truth_top_k=truth_hh, varepsilon=varepsilon,
                                        \
                    evaluate_type = evaluate_module_type, m=m, iterations=iterations)
                score_fedft += score
                
            score = score_fedft/runs
            scores.append(score)
            enablePrint()
        
        results.append(["FedFT", evaluate_module_type, *scores])
        
            
    pd.DataFrame(results, \
        columns= ["Method", "varepsilon",  0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7.5, 8.5, 9.5]
        ).to_csv(f"{save_path_dir}/heavyhitters_sentiment_inter.csv")
    pd.DataFrame(saved_csv_local_global, \
        columns=[f"cluster_{i}_lhh" for i in range(2)] + ["global_hh", "truth_ghh"]
        ).to_csv(f"{save_path_dir}/heavyhitters_sentiment_inter_local_global.csv")
    # with open(f"{CURR_DIR}/inter_cluster_exp_{evaluate_module_type}_m_{m}_k_{k}_iter_{iterations}.json", "w") as f:
    #     json.dump(results, f)
    
