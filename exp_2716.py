"""_summary_
This experiment used to compare different models when clients randomly loss connection.
Returns:
    _type_: _description_
"""
import os
import pickle
from server_FedFT import FedFTServer
from server import FAServerPEM
from server_WT import WTServer
from triehh import SimulateTrieHH
from Cipher import *
from utils import plot_all_in_one, load_clients




if __name__ == '__main__':
    n = 2000

    m = 64
    k = 8
    init_varepsilon = 5
    step_varepsilon = 0.2
    max_varepsilon = 9
    iterations = 32 

    round = 50

    connection_loss_rate = 0

    n = 1989 
    save_path_dir = f""  # result path 
    truth_top_k, clients = load_clients(filename=f"./dataset/zipf_remove_top5_{n}.txt", k=k)  # load clients from .txt

    results = {}
   
    while connection_loss_rate < 0.2:

        privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]
        evaluate_module_type = "F1" # ["NDCG", "F1"]

    #    # ----Standard Tree----( GRR + Uniform + Trie) #

    #     server = FAServerPEM(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
    #         privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate
    #     )
    
    #     x, y = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)

    #     if "PEM" not in results:
    #         results["PEM"] = {}
    #     results["PEM"][connection_loss_rate] = [x, y]
       
    #    # ---GTU----( GRR + Uniform + Trie) #

    #     server = WTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
    #         privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate
    #     )
    
    #     x, y = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)

    #     if "GTU" not in results:
    #         results["GTU"] = {}
    #     results["GTU"][connection_loss_rate] = [x, y]

        
        
        # ----GTF----( GRR + Trie + client_size_fitting ) #

        server = WTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate,
            is_uniform_size=False
        )
    
        x_gtf, y_gtf = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

        if "GTF" not in results:
            results["GTF"] = {}
        results["GTF"][connection_loss_rate] = [x_gtf, y_gtf]


    #     # ----GWU----( GRR + WT + uniform) #

    #     privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]
    #     server = WTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
    #         privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate,
    #         WT = True
    #     )
    
    #     x_gwu, y_gwu = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)

    #     if "GWU" not in results:
    #         results["GWU"] = {}
    #     results["GWU"][connection_loss_rate] = [x_gwu, y_gwu]

    #     # ----GWF----( GRR + WT + client_size_fitting) #

    #     privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]
    #     server = WTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
    #         privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate,
    #         is_uniform_size=False, WT=True
    #     )
    
    #     x_gwf, y_gwf = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)

    #     if "GWF" not in results:
    #         results["GWF"] = {}
    #     results["GWF"][connection_loss_rate] = [x_gwf, y_gwf]


        privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
        # ----XTU----( GRR_X + Trie + Uniform size) #
        # server = WTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
        #     privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate
        # )
    
        # x_xtu, y_xtu = server.server_run_plot_varepsilon(
        # init_varepsilon,  step_varepsilon, max_varepsilon)

        # if "XTU" not in results:
        #     results["XTU"] = {}
        # results["XTU"][connection_loss_rate] = [x_xtu, y_xtu]
         

    #    ----XTF----( GRR_X + Trie + client_size_fitting) #

        server = WTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k, \
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate,
            is_uniform_size=False
        )
    
        x_xtf, y_xtf = server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

        if "XTF" not in results:
            results["XTF"] = {}
        results["XTF"][connection_loss_rate] = [x_xtf, y_xtf]
        
        
    #     # ----Weight Tree---- (GRR_X + Uniform Client + WT) # 
    #     server = WTServer(n, m, k, init_varepsilon, iterations, round,clients=clients, C_truth=truth_top_k, 
    #     privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate)

    #     # server.server_run()
    #     xn, yn = server.server_run_plot_varepsilon(
    #         init_varepsilon,  step_varepsilon, max_varepsilon)

    #     if "WT" not in results:
    #         results["WT"] = {}
    #     results["WT"][connection_loss_rate] = [xn, yn]
        
 

        # ----FedFT (GRR_X + Client Size fitting + WT )---- # 
        server = WTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=truth_top_k,
            privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, connection_loss_rate=connection_loss_rate,
            WT=True, is_uniform_size=False)

        
        truth_top_k = server.C_truth[:]
        clients = server.clients[:]


        xc, yc = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)
        
        if "FedFT" not in results:
            results["FedFT"] = {}
        results["FedFT"][connection_loss_rate] = [xc, yc]

        

        connection_loss_rate += 0.1

    plot_all_in_one([x_xtf, x_gtf, xc], [y_xtf, y_gtf, yc], x_label="", y_label="", title="", labels=["XTF", "GTF", "XWF"])

    with open("./results/exp_WT.txt", 'wb') as f:
        pickle.dump(results, f)