
import json
from joblib import Parallel, delayed
import os, sys, random
import numpy as np
from _utils import get_non_iid_clusters_topk
from utils import blockPrint, enablePrint, encode_words
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
 
from tqdm import tqdm
from evaluate_module import EvaluateModule
from exp_generate_words import load_words, load_words_count

from server_FAServer import FA_aggregation
from intra_cluster_exp import DATA_PATH

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

saved_csv_local_global = []

def load_clusters(k=5):
    
    clusters_ = []
    
    cluster_top_k = []
    for n in range(2000, 10001, 1500):
        filename = f"words_generate_{n}"
        # cluster_truth_top_k = list(map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
        cluster = list(map(int,load_words(f"{DATA_PATH}{filename}_encode.txt")))
        
        clusters_.append(cluster)
        # cluster_top_k.append(cluster_truth_top_k)
        
    truth_hh = get_non_iid_clusters_topk(k) 
    truth_hh = encode_words(truth_hh)
    # truth_hh = get_bayes_shrinkage_topk(k)
    
    return clusters_, truth_hh

if __name__ == '__main__':

    m = 48
    k = 5

    
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
            score_FA_ = 0
            blockPrint() 
            for rnd in range(runs):
                score, saved_csv_local_global = FA_aggregation(clients, k, global_truth_top_k=truth_hh, \
                    saved_csv_local_global=saved_csv_local_global, varepsilon=varepsilon,   \
                    evaluate_type = evaluate_module_type, m=m, iterations=iterations)
                score_FA_ += score
            score = score_FA_/runs
            scores.append(score)
            enablePrint()
        
        results.append(["FA_", evaluate_module_type, *scores])
            
    import pandas as pd
    pd.DataFrame(results, \
        columns= ["Method", "varepsilon",  0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7.5, 8.5, 9.5]
        ).to_csv(f"{CURR_DIR}/inter_cluster_exp_{evaluate_module_type}_m_{m}_k_{k}_iter_{iterations}.csv")
    pd.DataFrame(saved_csv_local_global, \
        columns=[f"cluster_{i}_lhh" for i in range(len(range(2000, 10001, 1500)))] + ["global_hh", "truth_ghh"]
        ).to_csv(f"{CURR_DIR}/heavyhitters_inter_local_global.csv")
   
    
