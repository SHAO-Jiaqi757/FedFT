"""_summary_
This experiment used to compare different models in the intra-cluster setting.
On 6 datasets with 2000-9500 clients, seperatively.
"""
import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
 
from server import FAServerPEM
from server_AServer import Aserver
from Cipher import *
from utils import enablePrint, blockPrint
from exp_generate_words import load_words

enablePrint()
DATA_PATH = "dataset/words_generate/"



if __name__ == '__main__':

    save_path_dir = f""  # result path 
    m = 48
    k = 5
    
    # encode_file_initate(k)
    
    init_varepsilon = 0.5
    step_varepsilon = 1
    max_varepsilon = 9.6
    iterations = 24

    runs = 40
    run = False
    for n in range(2000, 10001, 1500):

        filename = f"words_generate_{n}"
        
        if run:
            blockPrint()
            truth_top_k = list(map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
            clients = list(map(int,load_words(f"{DATA_PATH}{filename}_encode.txt")))
            
            results = {}
        
            evaluate_module_type = "F1" # ["recall", "F1"]
            privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]

        # ----Standard Tree----( GRR + Uniform + Trie) #

            server = FAServerPEM(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
                privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type,  
            )
            x_pem, y_pem = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)

            results["PEM"] = [x_pem, y_pem]
            
            # ----GTF----( GRR + Trie + client_size_fitting ) #
            
            server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
                privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, 
                is_uniform_size=True
            )
        
            x_gtf, y_gtf = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)


            results["GTF"] = [x_gtf, y_gtf]


            privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
            # ----XTU----( GRR_X + Trie + Uniform size) #
            server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
                privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, 
                is_uniform_size=True
            )
        
            x_xtu, y_xtu = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)

            results["XTU"]= [x_xtu, y_xtu]
            
        #    ----FedFT----( GRR_X + Trie + client_size_fitting + optimization) #
        
            server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
                evaluate_type=evaluate_module_type,
                is_uniform_size=False)
        
            x_xtf, y_xtf = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)

            results["XTF"] = [x_xtf, y_xtf]
            
            with open(f"{save_path_dir}{filename}_result_{n}.txt", "w") as f:
                f.write(str(results))
        else:
            with open(f"{filename}_result_{n}.txt", "r") as f:
                results = eval(f.read())
            print(n)
            for result in results:
                print("\n" + result)
                x, y = results[result]
                # y = [a, b, c], print: a|b|..., and a, b ... rounded to 3 decimal places
                print("$\\varepsilon$ |" + "|".join([str(i) for i in x]) + "|")
                print("|----|----|----|----|----|----|----|----|----|----|----|")
                print("|recall |" + "|".join([str(round(i, 3)) for i in y])+ "|")

            print("\n\n")