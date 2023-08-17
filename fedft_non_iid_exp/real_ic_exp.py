"""_summary_
This experiment used to compare different models in the intra-cluster setting.
On 6 datasets with 2000-9500 clients, seperatively.
"""

import csv
import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
from exp_generate_words import load_words
from utils import enablePrint, blockPrint, encode_file_initate
from Cipher import *
from server_AServer import Aserver
from server import BaseServer

blockPrint()
DATA_PATH = "dataset/sentiment/"


if __name__ == '__main__':

    save_path_dir = f"plots/exp_non_iid/"  # result path
    m = 48
    k = 5


    init_varepsilon = 0.5
    step_varepsilon = 1
    max_varepsilon = 9.6
    iterations = 24

    runs = 20
    run = True
    results = [
        "No. Clients,Method,varepsilon,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5"]
    
    for _data_name in ["reddit", "sentiment"]:

        filename = f"heavyhitters_{_data_name}"

        if run:
            enablePrint()
            encode_file_initate(k, filename, DATA_PATH)
            
            truth_top_k = list(
                map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
            clients = list(
                map(int, load_words(f"{DATA_PATH}{filename}_encode.txt")))
            n = len(clients)
            blockPrint()
            for evaluate_module_type in ["recall", "F1"]:
                privacy_mechanism_type = "GRR"  # ["GRR", "None","OUE"]

            # ----Standard Tree----( GRR + Uniform + Trie) #

                server = BaseServer(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                     privacy_mechanism_type=privacy_mechanism_type, evaluate_type=evaluate_module_type,
                                     )
                x_pem, y_pem = server.server_run_plot_varepsilon(
                    init_varepsilon,  step_varepsilon, max_varepsilon)

                results.append(
                    f"{n}, PEM, {evaluate_module_type}, " + ",".join(map(str, y_pem)))

                # ----GTF----( GRR + Trie + client_size_fitting ) #

                server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                 privacy_mechanism_type=privacy_mechanism_type, evaluate_type=evaluate_module_type,
                                 is_uniform_size=True
                                 )

                x_gtf, y_gtf = server.server_run_plot_varepsilon(
                    init_varepsilon,  step_varepsilon, max_varepsilon)

                results.append(
                    f"{n}, GTF, {evaluate_module_type}, " + ",".join(map(str, y_gtf)))

                privacy_mechanism_type = "GRR_X"  # ["GRR", "None","OUE"]
                # ----XTU----( GRR_X + Trie + Uniform size) #
                server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                 privacy_mechanism_type=privacy_mechanism_type, evaluate_type=evaluate_module_type,
                                 is_uniform_size=True
                                 )

                x_xtu, y_xtu = server.server_run_plot_varepsilon(
                    init_varepsilon,  step_varepsilon, max_varepsilon)

                results.append(
                    f"{n}, XTU, {evaluate_module_type}, " + ",".join(map(str, y_xtu)))

            #    ----FedFT----( GRR_X + Trie + client_size_fitting + optimization) #

                server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k,
                                 evaluate_type=evaluate_module_type,
                                 is_uniform_size=False)

                x_xtf, y_xtf = server.server_run_plot_varepsilon(
                    init_varepsilon,  step_varepsilon, max_varepsilon)

                results.append(
                    f"{n}, XTF, {evaluate_module_type}, " + ",".join(map(str, y_xtf)))

                with open(f"{save_path_dir}{filename}_intra_{m}.csv", "w") as file:
                    writer = csv.writer(file)
                    for row in results:
                        writer.writerow(row.split(','))
        else:
            pass
            # with open(f"{filename}_result_{n}.txt", "r") as f:
            #     results = eval(f.read())
            # print(n)
            # for result in results:
            #     print("\n" + result)
            #     x, y = results[result]
            #     # y = [a, b, c], print: a|b|..., and a, b ... rounded to 3 decimal places
            #     print("$\\varepsilon$ |" + "|".join([str(i) for i in x]) + "|")
            #     print("|----|----|----|----|----|----|----|----|----|----|----|")
            #     print("|recall |" + "|".join([str(round(i, 3)) for i in y])+ "|")

            # print("\n\n")
