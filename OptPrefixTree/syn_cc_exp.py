#  

# add absolute path of current path's parent to import modules
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

CUR_DIR_NAME = sys.path[0].split('/')[-1] 

import numpy as np
import pickle
from triehh_non_iid_exp.preprocess import *
# from exp_sentiment_intra import evaluate, SimulateTrieHH
from evaluate_module import EvaluateModule
from exp_generate_words import load_words
from utils import enablePrint, blockPrint, encode_file_initate
from Cipher import *
from OptPrefixTree import OptPrefixTreeServer
DATA_PATH = "dataset/words_generate/"


if __name__ == '__main__':

    # load dictionary
    # please provide your own dictionary if you would like to
    # run out-of-vocabulary experiments
    m = 48
    k = 5

    results = []

    init_varepsilon = 5.5
    step_varepsilon = 1
    max_varepsilon = 9.6
    iterations = 12

    runs = 5 
  
    filename = f"words_generate_combined"
    
    encode_file_initate(k, filename, DATA_PATH)
    
    truth_top_k = list(
        map(int, load_words(f"{DATA_PATH}{filename}_encode_top_{k}.txt")))
    clients = list(
        map(int, load_words(f"{DATA_PATH}{filename}_encode.txt")))
    
    n = len(clients)
    f1_scores = []
    recall_scores = []
    # epsilon for differential privacy
    for evaluate_module_type in ["F1", "recall"]:
        server = OptPrefixTreeServer(n, m, k, init_varepsilon, iterations, 1, clients=clients, C_truth=truth_top_k,
                                    evaluate_type=evaluate_module_type
                                    ) # TOO TIME CONSUMING, change runs to 1
        x, y = server.server_run_plot_varepsilon(
            init_varepsilon,  step_varepsilon, max_varepsilon)
        if evaluate_module_type == "F1":
            f1_scores = y
        else:
            recall_scores = y
        results.append(
            f"OptPrefixTree, {evaluate_module_type}, " + ",".join(map(str, y)) )
                
            
        
    # save results
    with open(f"{filename}_result_{n}.txt", "w") as f:
        f.write(str(results))