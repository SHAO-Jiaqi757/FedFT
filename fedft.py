from math import tanh
import random
from time import sleep
import contextlib
import sys
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
from evaluate_module.evaluate_module import EvaluateModule
from server_AServer import Aserver
from Cipher import *
from utils import distance, load_clients, pr_N_mostFrequentNumber, encode_words


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


def fedft_cluster(clients: list, k: int, evaluate_module_type="recall", m=64, iterations=32, varepsilon=2, connection_loss_rate=0):
    """_summary_

    Args:
        clients (list): clients' data
        k (int): top k
        evaluate_module_type (str, optional): performace evaluation "recall", "F1" . Defaults to "recall".
        m (int, optional): maximum length of binary strings. Defaults to 64.
        iterations (int, optional): number of iterations/trie's height. Defaults to 32.
        varepsilon (float, optional): LDP privacy parameter. Defaults to 2.
    return:
        str: json string of one cluster's result
    """
    # encode clients into bitstrings
    round = 1

    # truth_top_k = pr_N_mostFrequentNumber(clients, k)

    # clients = encode_words(clients)

    # truth_top_k = encode_words(truth_top_k)

    n = len(clients)
    # print("[debug]:: n", n)
    server = Aserver(n, m, k, varepsilon, iterations, round, clients=clients, C_truth=[0],
                         evaluate_type=evaluate_module_type, connection_loss_rate=connection_loss_rate,
                         is_uniform_size=False, optimize=False
                         )

    x_xtf, _ = server.server_run()

    predict_hh = server.A_i

    results = {'predict_hh': predict_hh,  'eps': x_xtf}
    return json.dumps(results)



def fed_ft_aggregation(n: int, clients: list, k: int, global_truth_top_k: list, varepsilon: float = 2, evaluate_type="recall", cluster_size=5000, m=64, iterations=32, connection_loss_rate=0):
    """_summary_

    Args:
        n (int): number of clients
        clients (list): clients' data
        k (int): top k
        global_truth_top_k (list): global truth top k
        var_epsilon (float, optional): LDP privacy parameter. Defaults to 2.
        evaluate_type (str, optional): performace evaluation "recall", "F1" . Defaults to "recall".
        cluster_size (int, optional): number of clients in one cluster . Defaults to 5000.
        m (int, optional): maximum length of binary strings. Defaults to 64.
        iterations (int, optional): number of iterations/trie's height. Defaults to 32.

    Returns:
        float: score (F1 or recall)
    """
    random.shuffle(clients)

    clusters = n // cluster_size

    # Divide clients into clusters
    try:
        clients = np.reshape(clients, (clusters, cluster_size))
    except Exception:
        clients = clients[:clusters*cluster_size]
        clients = np.reshape(clients, (clusters, cluster_size))

    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        results = Parallel(n_jobs=clusters)(delayed(fedft_cluster)(
            clients[i], k, evaluate_type, m, iterations, varepsilon, connection_loss_rate) for i in tqdm(range(clusters)))

        noise_hhs = {}
        for i in range(clusters):
            result_i = json.loads(results[i])
            predict_hh = result_i['predict_hh']

            for hh in predict_hh:
                incre = tanh(0.2*k + predict_hh[hh])
                # incre = 1
                noise_hhs[hh] = noise_hhs.get(hh, 0) + incre

        # aggregate results
        noise_hhs = sorted(noise_hhs.items(),
                           key=lambda x: x[1], reverse=True)[: k * 4]
        noise_hhs_num = len(noise_hhs)
        good_hhs = []
        bad_hh = []
        # threshold = 5
        for i in range(noise_hhs_num):
            x = int(noise_hhs[i][0])
            if x in bad_hh:
                continue
            # threshold = (len(bin(x)) - 2)/4
            threshold = (len(bin(x)) - 2)/2
            for j in range(noise_hhs_num-1, i, -1):

                y = int(noise_hhs[j][0])

                dis = distance(x, y)
                if dis <= threshold and noise_hhs_num - len(bad_hh) > k:
                    bad_hh.append(y)
                    continue
                # print(f"{x} {y}: distance = {dis}")
            good_hhs.append(x)
            if len(good_hhs) == k: break
        # good_hhs = good_hhs[:k]
        print("global truth_top_k: ", global_truth_top_k)
        evaluate_module = EvaluateModule(evaluate_type)
        score = evaluate_module.evaluate(global_truth_top_k, good_hhs)
        print("finall hhs: ", good_hhs, " score:", score)

    print("Done!")
    return score


def fedft_running_rounds(n: int, clients: list, k: int, global_truth_top_k: list, varepsilon: float = 2, step_varepsilon=0.4, max_varepsilon=2.2,
                         evaluate_type="recall", cluster_size=5000, m=64, iterations=32, rounds=10, connection_loss_rate=0):
    """_summary_

    Args:
        n (int): number of clients
        clients (list): clients' data
        k (int): top k
        global_truth_top_k (list): global truth top k
        varepsilon (float, optional): LDP privacy parameter. Defaults to 2.
        step_varepsilon (float, optional): step of varepsilon. Defaults to 0.4.
        max_epsilon (float, optional): max varepsilon. Defaults to 2.2.
        cluster_size (int, optional): number of clients in one cluster . Defaults to 5000.
        m (int, optional): maximum length of binary strings. Defaults to 64.
        iterations (int, optional): number of iterations/trie's height. Defaults to 32.
        rounds (int, optional): number of rounds. Defaults to 10.

    Returns:
        list[float], list[float]: epsilons, scores
    """
    varepsilons = []
    scores = []
    while varepsilon < max_varepsilon:
        # running rounds
        score = 0
        for i in range(rounds):
            score += fed_ft_aggregation(n, clients, k, global_truth_top_k, varepsilon, evaluate_type, cluster_size, m, iterations, connection_loss_rate)
                  
        varepsilons.append(varepsilon)
        scores.append(score/rounds)
        varepsilon += step_varepsilon

    # average score
    return varepsilons, scores
