"""_summary_
This experiment used to compare different models when clients randomly loss connection.
Returns:
    _type_: _description_
"""
import os
import pickle
from server_FedFT import FedFTServer
from server import FAServerPEM
from server_tree_weighted import FAServer
from triehh import SimulateTrieHH
from Cipher import *
from utils import plot_all_in_one, load_clients




if __name__ == '__main__':
    n = 2000

    m = 64
    k = 8
    init_varepsilon = 0.2
    step_varepsilon = 0.8
    max_varepsilon = 12
    iterations = 32

    round = 10

    connection_loss_rate = 0

    # client_size = 2716
    # save_path_dir = f"./results/connectionloss_{client_size}"
    # truth_top_k, clients = load_clients(filename=f"./dataset/triehh_clients_remove_top5_{client_size}.txt", k=k)

    save_path_dir = ''
    truth_top_k =[]
    clients = []
    client_distribution_type = ""
    while connection_loss_rate < 0.7:
        privacy_mechanism_type = "GRR_Weight" # ["GRR", "None","OUE"]
        evaluate_module_type = "F1" # ["NDCG", "F1"]

        # ----Weight Tree & Client Size fitting---- # 
        server = FedFTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k,
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate)

        truth_top_k = server.C_truth[:]
        clients = server.clients[:]
        if not client_distribution_type:
            client_distribution_type = server.client_distribution_type
        save_path_dir = f'./results/{client_distribution_type}_{1.5}_{server.n}'

        xc, yc = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)
        
        with open(os.path.join(save_path_dir, f"fedft_cls{connection_loss_rate:.1f}_{evaluate_module_type}"), 'wb') as f:
            pickle.dump([xc, yc], f)
        # server.server_run()
    
         # ----Weight Tree---- # 
        server = FAServer(n, m, k, init_varepsilon, iterations, round,clients=clients, C_truth=truth_top_k, 
        privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate)

        # server.server_run()
        xn, yn = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)
        with open(os.path.join(save_path_dir, f"wtrie_cls{connection_loss_rate:.1f}_{evaluate_module_type}"), 'wb') as f:
            pickle.dump([xn, yn], f) 
        # ----Standard Tree---- #

        privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]
        server = FAServerPEM(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate
        )
    
        x, y = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)
        with open(os.path.join(save_path_dir, f"pem_cls{connection_loss_rate:.1f}_{evaluate_module_type}"), 'wb') as f:
            pickle.dump([x, y], f) 

        # ----TrieHH Tree---- #
        delta = 1/(n**2)
        evaluate_module_type = "F1" # ["NDCG", "F1"]

        server = SimulateTrieHH(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k,  
                delta=delta, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate)
        # server.server_run()
        x_triehh, y_triehh = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)

        with open(os.path.join(save_path_dir, f"triehh_cls{connection_loss_rate:.1f}_{evaluate_module_type}"), 'wb') as f:
            pickle.dump([x_triehh, y_triehh], f) 
        
        # Visualize Comparison ##
        xs = [xc, xn, x, x_triehh]
        ys = [yc, yn, y, y_triehh]

        plot_all_in_one(xs, ys, "privacy budget", "F1", f"Connection loss rate={connection_loss_rate:.1f}", [ "FedFT", "Weight Trie", "PEM", "TrieHH"] )

        connection_loss_rate += 0.1