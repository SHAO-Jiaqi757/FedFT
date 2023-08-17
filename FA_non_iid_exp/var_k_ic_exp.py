"""_summary_
This experiment used to compare different models in the intra-cluster setting.
On 6 datasets with 2000-9500 clients, seperatively.
"""

import pandas as pd
import sys, os
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
from exp_generate_words import load_words
from utils import enablePrint, blockPrint
from Cipher import *
from server_FAServer import FAserver
from server import BaseServer

enablePrint()
DATA_PATH = "dataset/words_generate/"

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    m = 48
    
    varepsilon = 4.5
    min_k = 6
    max_k = 25
    step_k = 6 
    iterations = 24
    runs = 20 
    
    run = True
    results = []
    
    n = 9500

    for k in range(min_k, max_k, step_k):

        filename = f"words_generate_{n}"

        if run:
            blockPrint()

            truth_top_k = list(
                map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
            clients = list(
                map(int, load_words(f"{DATA_PATH}{filename}_encode.txt")))
            for evaluate_module_type in ["recall", "F1"]:
                privacy_mechanism_type = "GRR"  # ["GRR", "None","OUE"]

            # ----Standard Tree----( GRR + Uniform + Trie) #

                server = BaseServer(n, m, k, varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                     privacy_mechanism_type=privacy_mechanism_type, evaluate_type=evaluate_module_type,
                                     )
                x_pem, y_pem = server.server_run()

                results.append(
                    ["PEM", k, evaluate_module_type,varepsilon, y_pem])

                # ----GTF----( GRR + Trie + client_size_fitting ) #

                server = FAserver(n, m, k, varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                 privacy_mechanism_type=privacy_mechanism_type, evaluate_type=evaluate_module_type,
                                 is_uniform_size=True
                                 )

                x_gtf, y_gtf = server.server_run()

                results.append(
                    ["GTF", k, evaluate_module_type,varepsilon, y_gtf])

                privacy_mechanism_type = "GRR_X"  # ["GRR", "None","OUE"]
                # ----XTU----( GRR_X + Trie + Uniform size) #
                server = FAserver(n, m, k, varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                 privacy_mechanism_type=privacy_mechanism_type, evaluate_type=evaluate_module_type,
                                 is_uniform_size=True
                                 )

                x_xtu, y_xtu = server.server_run()

                results.append(
                    ["XTU", k, evaluate_module_type,varepsilon, y_xtu])

            #    ----FA_----( GRR_X + Trie + client_size_fitting + optimization) #

                server = FAserver(n, m, k, varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                 evaluate_type=evaluate_module_type,
                                 is_uniform_size=False)

                x_xtf, y_xtf = server.server_run()

                results.append(
                    ["XTF", k, evaluate_module_type,varepsilon, y_xtf])

                pd.DataFrame(results,
                             columns=["Method", "k", "metrics", "epsilon", "metric_value"]).to_csv(f'{CURR_DIR}/intra_{n}_var_k_{min_k}_{max_k}.csv')
        else:
            pass
            
