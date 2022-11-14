import random
from time import sleep
import contextlib
import sys
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
import os
import pickle
from evaluate_module.evaluate_module import EvaluateModule
from server_FAServer import FedFTServer
from Cipher import *
from utils import distance, load_clients, pr_N_mostFrequentNumber,encode_words


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

def fedft_group(clients, k, evaluate_module_type="recall"):
    # encode clients into bitstrings
    m = 64
    init_varepsilon = 2
    step_varepsilon = 0.4
    max_varepsilon = 2.2
    iterations = 32 

    round = 1
    
    # truth_top_k = pr_N_mostFrequentNumber(clients, k)
    
    # clients = encode_words(clients)
    
    # truth_top_k = encode_words(truth_top_k)
    
    n = len(clients)
    # print("[debug]:: n", n)
    server = FedFTServer(n, m, k, init_varepsilon, iterations, round, clients=clients, C_truth=[0], \
        evaluate_type=evaluate_module_type,
    is_uniform_size=False, optimize=True
)
    
    x_xtf, y_xtf = server.server_run_plot_varepsilon(
    init_varepsilon,  step_varepsilon, max_varepsilon)
    
    predict_hh = server.A_i
    
    results = {'predict_hh': predict_hh,  'eps': x_xtf, 'score': y_xtf}
    return json.dumps(results)



def fed_ft_main(n, clients, k, global_truth_top_k, evaluate_type="recall", cluster_size=5000):

    random.shuffle(clients)

    clusters = n // cluster_size

    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    with std_out_err_redirect_tqdm() as orig_stdout:
        results = {}
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        for i in tqdm(range(clusters), file=orig_stdout, dynamic_ncols=True ):
            sleep(.5)
            clients_ = clients[i*cluster_size:(i+1)*cluster_size]
            out = fedft_group(clients_, k)
            results[i] = out

        global_pred_hhs = {}
        for i in range(clusters):
            result_i = json.loads(results[i])
            predict_hh = result_i['predict_hh']
            score = result_i['score']
            
            for hh in predict_hh:
                global_pred_hhs[hh] = global_pred_hhs.get(hh, 0) + 1
        
        
        # aggregate results
        # global_pred_hhs = sorted(global_pred_hhs.items(), key=lambda x: x[1], reverse=True)[:k]
        global_pred_hhs = sorted(global_pred_hhs.items(), key=lambda x: x[1], reverse=True)[: k * 4]
        
        # m = k*4
        # for indx in range(len(global_pred_hhs)):
        #     if global_pred_hhs[indx][1] < clusters/k or indx >= m:
        #         break 
        # global_pred_hhs = global_pred_hhs[:indx]
            
        good_hhs = []
        noise_hh = []
        threshold = 5
        for i in range(len(global_pred_hhs)):
            x = int(global_pred_hhs[i][0])
            if x in noise_hh:
                continue                
            for j in range(i+1, len(global_pred_hhs)):

                y = int(global_pred_hhs[j][0])

                dis = distance(x, y)
                if dis <= threshold:
                    noise_hh.append(y)
                    continue
                # print(f"{x} {y}: distance = {dis}")
            good_hhs.append(x)
        good_hhs = good_hhs[:k]
        print("global truth_top_k: ", global_truth_top_k)
        evaluate_module = EvaluateModule(k, evaluate_type)
        score = evaluate_module.evaluate(global_truth_top_k, good_hhs)
        print("finall hhs: ", good_hhs, " score:", score)
        
    print("Done!")
    return k, score
