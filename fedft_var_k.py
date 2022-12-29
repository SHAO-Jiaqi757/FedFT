"""_summary_
This experiment used to compare different models when clients randomly loss connection.
Returns:
    _type_: _description_
"""
import os
import pickle
from time import sleep
from tqdm import tqdm
from group_fedft import fed_ft_aggregation
from server import FAServerPEM
from server_FAServer import FedFTServer
from triehh import SimulateTrieHH
from Cipher import *
from utils import blockPrint,enablePrint, plot, load_clients, pr_N_mostFrequentNumber, encode_word




if __name__ == '__main__':

    # blockPrint()
    enablePrint()
    m = 64
    k = 6
    init_varepsilon = 2
    step_varepsilon = 0.4
    max_varepsilon = 2.2
    iterations = 32 

    round = 20 

    # n = 99413
    # n = 659596 
    cluster_n = 5000
    # origin_clients_file = f"./dataset/sentiment140_user_clean.txt"
    # encoded_clients_file = f"./dataset/sentiment140_user_clean_encode.txt"
    # topk_words_file = f"./dataset/sentiment140_user_clean_topk.txt"
    # save_filename = f"./results/sentiment140_user_clean_var_k"
    origin_clients_file = f"./dataset/Reddit_clean.txt"
    encoded_clients_file = f"./dataset/Reddit_clean_encode.txt"
    topk_words_file = f"./dataset/Reddit_clean_topk.txt"
    save_filename = f"./results/Reddit_clean_var_k"
    # n = 100000
    # origin_clients_file = f"./dataset/zipf_clean_{n}.txt"
    # encoded_clients_file = f"./dataset/zipf_clean_{n}_encode.txt"
    # topk_words_file = f"./dataset/zipf_clean_{n}_topk.txt"
    # save_filename = f"./results/zipf_clean_{n}_var_k"

    
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

    results = {}
   
    while k  < 30:
        truth_top_k = sorted_words[:k]
        for i in range(len(truth_top_k)):
            number = encode_word(truth_top_k[i])
            truth_top_k[i] = number 
            
        evaluate_module_type = "recall" # ["recall", "F1"]
        privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]

       # ----Standard Tree----( GRR + Uniform + Trie) #

        server = FAServerPEM(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type
        )
        _, y_pem = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)
        
        print(y_pem)

        if "PEM" not in results:
            results["PEM"] = {'x': [], 'y': []}
        results["PEM"]['x'].append(k)
        results["PEM"]['y'].append(y_pem[0])
         

    #    ----FedFT----( GRR_X + Trie + client_size_fitting + optimization) #
    
        # server = FedFTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
        #       evaluate_type=evaluate_module_type,
        #     is_uniform_size=False, optimize=False
        # )
    
        # _, y_xtf = server.server_run_plot_varepsilon(
        # init_varepsilon,  step_varepsilon, max_varepsilon)
        # score = y_xtf[0]
        
        score_fedft = 0
        for rnd in range(round):
            score = fed_ft_aggregation(n, clients, k, global_truth_top_k=truth_top_k, evaluate_type = evaluate_module_type, cluster_size=cluster_n)
            score_fedft += score
        score = score_fedft/round

        if "FedFT" not in results:
            results["FedFT"] = {'y': [], 'x': []}
        results["FedFT"]['x'].append(k)
        results["FedFT"]['y'].append(score)
        
    
    # ---- TrieHH ----
        server = SimulateTrieHH(n, m, k, init_varepsilon, iterations, round, \
        clients=clients, C_truth=truth_top_k,
            delta= 1/(n**2), evaluate_type=evaluate_module_type)
        _, y_triehh = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)
        
        if "TrieHH" not in results:
            results["TrieHH"] = {"x": [], "y": []}
        # results["TrieHH"][k] = y_triehh
        results["TrieHH"]['x'].append(k)
        results["TrieHH"]['y'].append(y_triehh[0])
        

        k += 3
        

    try:
        with open(save_filename+".txt", 'rb') as f:
            res = pickle.load(f)
           
        results['PEM'] = res["PEM"]
        results["TrieHH"] = res["TrieHH"]
        
        results["FedFT"] = res["FedFT"]
    except Exception: # save results
        with open(save_filename+".txt", 'wb') as f:
            pickle.dump(results, f)  

    finally:        
        with open(save_filename+".txt", 'wb') as f:
            pickle.dump(results, f) 

    plot(results, save_filename=save_filename, x_label='k', y_label=evaluate_module_type, line_type="*-")

    