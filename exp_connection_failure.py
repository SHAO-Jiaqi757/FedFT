"""_summary_
This experiment used to compare different models when clients randomly loss connection.
Returns:
    _type_: _description_
"""
import os
import pickle
from exp_mode_query import FedFTServer
from server import FAServerPEM
from server_FAServer import FAServer
from triehh import SimulateTrieHH
from Cipher import *
from utils import plot_all_in_one, load_clients




if __name__ == '__main__':
    n = 2000

    m = 64
    k = 8
    init_varepsilon = 0.5
    step_varepsilon = 0.6
    max_varepsilon = 9
    iterations = 32 

    round = 50

    connection_loss_rate = 0

    n = 99413
    save_path_dir = f""  # result path 
    truth_top_k, clients = load_clients(filename=f"./dataset/zipf_{n}.txt", k=k)  # load clients from .txt

    results = {}
   
    while connection_loss_rate < 0.2:

        privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]
        evaluate_module_type = "F1" # ["NDCG", "F1"]

       # ----Standard Tree----( GRR + Uniform + Trie) #

        server = FAServerPEM(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate
        )
    
        x_pem, y_pem = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

        if "PEM" not in results:
            results["PEM"] = {}
        results["PEM"][connection_loss_rate] = [x_pem, y_pem]
       
        
        
        # ----GTF----( GRR + Trie + client_size_fitting ) #
        
        server = FAServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate,
            is_uniform_size=False
        )
    
        x_gtf, y_gtf = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

        if "GTF" not in results:
            results["GTF"] = {}
        results["GTF"][connection_loss_rate] = [x_gtf, y_gtf]



        privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
        # ----XTU----( GRR_X + Trie + Uniform size) #
        server = FAServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate
        )
    
        x_xtu, y_xtu = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

        if "XTU" not in results:
            results["XTU"] = {}
        results["XTU"][connection_loss_rate] = [x_xtu, y_xtu]
         

    #    ----XTF----( GRR_X + Trie + client_size_fitting) #

        server = FAServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate,
            is_uniform_size=False
        )
    
        x_xtf, y_xtf = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

        if "XTF" not in results:
            results["XTF"] = {}
        results["XTF"][connection_loss_rate] = [x_xtf, y_xtf]
        
   

        connection_loss_rate += 0.1

    # plot_all_in_one([x_pem, x_gtf, x_xtu, x_xtf], [y_pem, y_gtf, y_xtu, y_xtf], x_label="", y_label="", title="", labels=["PEM(GTU)", "GTF", "XTU", "PEM(XTF)"])

    with open("./results/exp_WT.txt", 'wb') as f:
        pickle.dump(results, f)