
import json
from joblib import Parallel, delayed
import os, sys
from _utils import load_clients, get_non_iid_clusters_topk
from utils import blockPrint, enablePrint
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
 
from tqdm import tqdm
from exp_generate_words import load_words

from FA_ import FA_cluster, std_out_err_redirect_tqdm, distance

DATA_PATH ="dataset/steps/"
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def FA_aggregation(clients: list, k: int, varepsilon: float = 2, m=64, iterations=32, stop_iter=-1):
    """_summary_

    Args:
        clients (list[list]): list of clusters of clients
        k (int): top k
        varepsilon (float, optional): LDP privacy parameter. Defaults to 2.
        m (int, optional): maximum length of binary strings. Defaults to 64.
        iterations (int, optional): number of iterations/trie's height. Defaults to 32.

    Returns:
        [list]: list of top k
    """
    
    clusters = len(clients)
    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        results = Parallel(n_jobs=clusters)(delayed(FA_cluster)(
            clients[i], k, m=m, iterations=iterations, varepsilon=varepsilon, stop_iter=stop_iter ) for i in tqdm(range(clusters)))

    candidates_among_clusters = {}
        
    for i in range(len(results)):
        result_i = json.loads(results[i])
        predict_hh:dict = result_i['predict_hh']
        threshold = k if k < list(predict_hh.values())[0] else 0
        for hh in predict_hh:
            x = predict_hh[hh] 
            if x <= threshold:
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
    return good_hhs

def load_clusters(k=5):
    
    clusters_ = []
    
    for n in range(5):
        filename = f"week_{n}"
        cluster = list(map(int,load_words(f"{DATA_PATH}{filename}.txt")))
        
        clusters_.append(cluster)
            
    return clusters_

if __name__ == '__main__':

    m = 48
    k = 1
    
    varepsilon = 2.5

    iterations = 24
    stop_iter = 18

    runs = 20

    clients = load_clusters(k) 
    results = {}

    
    cumulate = 0
    enablePrint()
    for rnd in range(runs):
        cumulate += FA_aggregation(clients, k, varepsilon=varepsilon,\
             m=m, iterations=iterations, stop_iter=stop_iter)[0]
    print(cumulate/runs)
    
# with open(f"{CURR_DIR}/steps.json", "w") as f:
#     json.dump(results, f)
    
