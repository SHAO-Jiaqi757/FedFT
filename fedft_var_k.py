"""_summary_
This experiment used to compare different models when clients randomly loss connection.
Returns:
    _type_: _description_
"""
import os
import pickle
import pandas as pd
from fedft import fed_ft_aggregation
from triehh_non_iid_exp import * 
from Cipher import *
from triehh_non_iid_exp.main import evaluate
from utils import blockPrint,enablePrint, plot, load_clients, pr_N_mostFrequentNumber, encode_word


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    blockPrint()
    # enablePrint()
    m = 48 
    min_k = 6
    max_k = 37
    epsilon = 2
    num_runs = 20 
    
    # length of longest word
    max_word_len = 10
    # delta for differential privacy
    delta = 2.3e-12
    cluster_n = 5000
    origin_clients_file = f"./dataset/Reddit_clean.txt"
    encoded_clients_file = f"./dataset/Reddit_clean_encode.txt"
    topk_words_file = f"./dataset/Reddit_clean_topk.txt"
    save_filename = f"./results/Reddit_clean_var_k"

    client_path = generate_triehh_clients("./dataset/Reddit_clean.txt")  
    word_counts = load_words_count("./dataset/Reddit_clean_count.txt")
    
    with open(client_path, 'rb') as fp:
      clients_top_word = pickle.load(fp)

    try:
        with open(encoded_clients_file, "rb") as f:
            clients = pickle.load(f)
        with open(topk_words_file,"rb") as f:
            sorted_words = pickle.load(f)
    except Exception:
        with open(origin_clients_file, "rb") as f:
            clients = pickle.load(f)
        # print(clients[:10]) 
        sorted_words = pr_N_mostFrequentNumber(clients)
        with open(topk_words_file, "wb") as f:
            pickle.dump(sorted_words, f)
            
        # encode words
        for i, word in enumerate(clients):
            number = encode_word(word)
            clients[i] = number
            # print(number)
        with open(encoded_clients_file, 'wb') as f:
            pickle.dump(clients, f)
        
    n = len(clients)

    results = []
    for evaluate_module_type in ["F1", "recall"]: 
        for k in range(min_k, max_k, 6):
            truth_top_k = sorted_words[:k]
            for i in range(len(truth_top_k)):
                number = encode_word(truth_top_k[i])
                truth_top_k[i] = number 
                
            evaluate_module_type = "recall" # ["recall", "F1"]
            
            score_fedft = 0
            for rnd in range(num_runs):
                score = fed_ft_aggregation(n, clients, k, varepsilon=epsilon, global_truth_top_k=truth_top_k, evaluate_type = evaluate_module_type, cluster_size=cluster_n)
                score_fedft += score
            score = score_fedft/num_runs

            results.append(["FedFT", k, evaluate_module_type, epsilon, score])

        # ---- TrieHH ----
            truth_triehh = list(word_counts.keys())[:k]

            simulate_triehh = SimulateTrieHH(client_path, max_word_len=10, epsilon=epsilon, delta=delta, num_runs=num_runs)
            triehh_heavy_hitters = simulate_triehh.get_heavy_hitters()
            
            score_triehh, _ = evaluate(evaluate_module_type, triehh_heavy_hitters, truth_triehh)
            results.append(["TrieHH", k, evaluate_module_type, epsilon, score_triehh])

            
            
    pd.DataFrame(results,
                 columns=["Method", "k", "metrics", "epsilon", "metric_value"]).to_csv(f"{CURR_DIR}/inter_var_k_{min_k}_{max_k}_epsilon_{epsilon}.csv")
        

            
            
            
            
            




        