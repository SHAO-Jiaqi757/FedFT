"""_summary_
This experiment used to compare different models in the intra-cluster setting.
On 6 datasets with 2000-9500 clients, seperatively.
"""
import os, sys, random
import pickle
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
 
from server import FAServerPEM
from server_AServer import Aserver
from Cipher import *
from utils import enablePrint, blockPrint
from exp_generate_words import load_words, load_words_count

enablePrint()
# blockPrint()
DATA_PATH = "dataset/words_generate/"


def load_clients(clients: list, top_k_words: list, k = sys.maxsize, restriction=-1, encode=True):

    if not encode:
        return top_k_words, clients
        
    for i in range(len(clients)):
        number = encode_word(clients[i])
        clients[i] = number 
    for i in range(len(top_k_words)):
        word = top_k_words[i]
        number = encode_word(word)
        top_k_words[i] = number 
     
    return top_k_words, clients


def encode_file_initate():
    for n in range(2000, 10001, 1500):
        filename = f"words_generate_{n}"
        
        encode_words(filename, 5)


def encode_words(filename, k: int):
    
    client_path = f" {DATA_PATH}{filename}.txt"
    freq_path = f" {DATA_PATH}{filename}_count.txt"
    
    words = load_words(client_path)
    word_counts = load_words_count(freq_path, k)
    top_k_words = list(word_counts.keys())
    top_k_encode_words, encode_words = load_clients(words, top_k_words=top_k_words, encode=True)
    
    with open(f" {DATA_PATH}{filename}_encode.txt", "w") as f:
        for word in encode_words:
            f.write(str(word) + " ")
            
    with open(f" {DATA_PATH}{filename}_encode_top_{k}.txt", "w") as f:
        for word in top_k_encode_words:
            f.write(str(word) + " ")



if __name__ == '__main__':

    save_path_dir = f""  # result path 
    m = 32
    k = 5
    
    init_varepsilon = 0.5
    step_varepsilon = 1 
    max_varepsilon = 9.6
    iterations = 16

    runs = 20

    for n in range(2000, 10001, 1500):
        filename = f"words_generate_{n}"
    #     truth_top_k = list(map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
    #     clients = list(map(int,load_words(f"{DATA_PATH}{filename}_encode.txt")))
        
    #     results = {}
    
    #     evaluate_module_type = "recall" # ["recall", "F1"]
    #     privacy_mechanism_type = "GRR" # ["GRR", "None","OUE"]

    # # ----Standard Tree----( GRR + Uniform + Trie) #

    #     server = FAServerPEM(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
    #         privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type,  
    #     )
    #     x_pem, y_pem = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)

    #     results["PEM"] = [x_pem, y_pem]
        
    #     # ----GTF----( GRR + Trie + client_size_fitting ) #
        
    #     server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
    #         privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, 
    #         is_uniform_size=False, optimize=False
    #     )
    
    #     x_gtf, y_gtf = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)


    #     results["GTF"] = [x_gtf, y_gtf]


    #     privacy_mechanism_type = "GRR_X" # ["GRR", "None","OUE"]
    #     # ----XTU----( GRR_X + Trie + Uniform size) #
    #     server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
    #         privacy_mechanism_type = privacy_mechanism_type, evaluate_type = evaluate_module_type, 
    #         is_uniform_size=True
    #     )
    
    #     x_xtu, y_xtu = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)

    #     results["XTU"]= [x_xtu, y_xtu]
        
    # #    ----FedFT----( GRR_X + Trie + client_size_fitting + optimization) #
    
    #     server = Aserver(n, m, k, init_varepsilon, iterations, runs, clients=clients, C_truth=truth_top_k, \
    #          evaluate_type=evaluate_module_type,
    #         is_uniform_size=False)
    
    #     x_xtf, y_xtf = server.server_run_plot_varepsilon(
    #     init_varepsilon,  step_varepsilon, max_varepsilon)

    #     results["XTF"] = [x_xtf, y_xtf]

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