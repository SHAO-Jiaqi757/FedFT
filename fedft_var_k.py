"""_summary_
This experiment used to compare different models when clients randomly loss connection.
Returns:
    _type_: _description_
"""
import os
import pickle

import numpy as np
from exp_mode_query import FedFTServer
from server import FAServerPEM
from server_FAServer import FedFTServer
from triehh import SimulateTrieHH
from Cipher import *
from utils import plot, load_clients, pr_N_mostFrequentNumber




if __name__ == '__main__':

    m = 64
    k = 3
    init_varepsilon = 3
    step_varepsilon = 0.4
    max_varepsilon = 3.2
    iterations = 32 

    round = 20

    # n = 99413
    n = 5000
    save_path_dir = f""  # result path 
    with open(f"./dataset/zipf_clean_{n}.txt", "rb") as f:
        clients = pickle.load(f)
        
    sorted_words = pr_N_mostFrequentNumber(clients)
            
    for i in range(len(clients)):
        number = encode_word(clients[i])
        clients[i] = number 

     
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
         
        # # ----GTF----( GRR + Trie + client_size_fitting ) #
        
        # server = FedFTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
        #     privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type,  
        #     is_uniform_size=False, optimize=False
        # )
    
        # _, y_gtf = server.server_run_plot_varepsilon(
        # init_varepsilon,  step_varepsilon, max_varepsilon)

        # if "GTF" not in results:
        #     results["GTF"] = {'x':[], 'y':[]}
        # results["GTF"]['x'].append(k)
        # results["GTF"]['y'].append(y_gtf[0])


        # privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
        # # ----XTU----( GRR_X + Trie + Uniform size) #
        # server = FedFTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
        #     privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type,  
        #     is_uniform_size=True, optimize=False
        # )
    
        # _, y_xtu = server.server_run_plot_varepsilon(
        # init_varepsilon,  step_varepsilon, max_varepsilon)

        # if "XTU" not in results:
        #     results["XTU"] = {'x': [], 'y': []}
        # results["XTU"]['x'].append(k)
        # results["XTU"]['y'].append(y_xtu[0])
         

    #    ----FedFT----( GRR_X + Trie + client_size_fitting + optimization) #
    
        server = FedFTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
              evaluate_type=evaluate_module_type,
            is_uniform_size=False, optimize=False
        )
    
        _, y_xtf = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

        if "FedFT(XTF)" not in results:
            results["FedFT(XTF)"] = {'y': [], 'x': []}
        results["FedFT(XTF)"]['x'].append(k)
        results["FedFT(XTF)"]['y'].append(y_xtf[0])
        
    
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

    # plot_all_in_one([x_pem, x_gtf, x_xtu, x_xtf], [y_pem, y_gtf, y_xtu, y_xtf], x_label="", y_label="", title="", labels=["PEM(GTU)", "GTF", "XTU", "PEM(XTF)"])

    save_filename = f"./results/zipf_clean_{n}_var_k"
        
    plot(results, save_filename=save_filename, x_label='k', y_label=evaluate_module_type, line_type="*-")

    