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
    init_varepsilon = 3.7
    step_varepsilon = 0.6
    max_varepsilon = 3.0
    iterations = 32 

    round = 50

    n = 99413
    save_path_dir = f""  # result path 
    truth_top_k, clients = load_clients(filename=f"./dataset/zipf_{n}.txt", k=k)  # load clients from .txt

    results = {}
   
    for iterations in [64, 32, 16, 8]:

    #    ----XTF----( GRR_X + Trie + client_size_fitting) #

        server = FAServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = "GRR_X", evaluate_type = "F1",
            is_uniform_size=False
        )
        print(f"[FedFT]:: bits_per_iter = {server.bits_per_iter}")
        x_xtf, y_xtf = server.server_run()
        
        if "XTF" not in results:
            results["XTF"] = {}
        results["XTF"][server.bits_per_iter] = [x_xtf, y_xtf]
         


    # plot_all_in_one([x_pem, x_gtf, x_xtu, x_xtf], [y_pem, y_gtf, y_xtu, y_xtf], x_label="", y_label="", title="", labels=["PEM(GTU)", "GTF", "XTU", "PEM(XTF)"])

    with open("./results/exp_bits.txt", 'wb') as f:
        pickle.dump(results, f)