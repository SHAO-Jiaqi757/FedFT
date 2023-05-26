from math import exp, log, sqrt
import os
import pickle
import sys
import random
import time
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from Cipher import *
from pathlib import Path
import pandas as pd

random.seed(0)

def sort_by_frequency(array, k=sys.maxsize) -> list:
    return np.argsort(-np.bincount(array))[:k]


def load_words(file_path):
    with open(file_path, "r") as f:
        words = f.read().strip().split(" ")
    return words

def load_words_count(file_path, top_k=-1):
    # file {word: count} sorted by count
    with open(file_path, "r") as f:
        # top_k lines
        words = f.readlines()
        if top_k != -1:
            words = words[:top_k]
        words = [word.split(":") for word in words]
        words = {word[0]: int(word[1]) for word in words}
    
    return words


def encode_file_initate(k=5, filename="", datadir=""):
    # def test():
    #     print("test")
    #     freq_path = f"{datadir}{filename}_count.txt"
    #     save_path_encoded = f"{datadir}{filename}_encode.txt"
    #     word_counts = load_words_count(freq_path, 10)
    #     test_10 = f"{datadir}{filename}_remove10_encode.txt"
    #     unique_words, unique_word_counts = list(word_counts.keys()), list(word_counts.values())
        
    #     for i in range(len(unique_words)):
    #         word = unique_words[i]
    #         number = encode_word(word)
    #         unique_words[i] = number  
        
    #     # copy test_10 to save_path_encoded 
    #     with open(test_10, "r") as f:
    #         words = f.read().strip().split(" ")
    #     for word, word_count in zip(unique_words, unique_word_counts):
    #         words += [str(word)] * word_count
    #     with open(save_path_encoded, "w") as f:
    #         f.write(" ".join(words))

    def encode_words(filename, k: int, datadir) :
        
        freq_path = f"{datadir}{filename}_count.txt"
        word_counts = load_words_count(freq_path)
        save_path_encoded = f"{datadir}{filename}_encode.txt"
        save_path_encoded_topk =f"{datadir}{filename}_encode_top_{k}.txt" 
        if os.path.exists(save_path_encoded) and os.path.exists(save_path_encoded_topk):
            print("file already exist")
            return 
        # if file f"{datadir}{filename}_encode.txt" not exist
        if not os.path.exists(save_path_encoded):
            unique_words, unique_word_counts = list(word_counts.keys()), list(word_counts.values())
        else: 
            unique_words, unique_word_counts = list(word_counts.keys())[:k], list(word_counts.values())[:k] 
        
        for i in range(len(unique_words)):
            word = unique_words[i]
            number = encode_word(word)
            unique_words[i] = number 
        top_k_encode_words = unique_words[:k]
        
        if not os.path.exists(save_path_encoded):
            encode_words = []
            for word, word_count in zip(unique_words, unique_word_counts):
                encode_words += [word] * word_count
           
            with open(f"{datadir}{filename}_encode.txt", "w") as f:
                for word in encode_words:
                    f.write(str(word) + " ")
            print("encode file saved")
        if not os.path.exists(save_path_encoded_topk):      
            with open(save_path_encoded_topk, "w") as f:
                for word in top_k_encode_words:
                    f.write(str(word) + " ")
            print("top k encode file saved")

    encode_words(filename, k, datadir)

def visualize_frequency(array, truth_top_k_heavy_hitters, distribution_type, k=sys.maxsize):
    plt.figure()
    plt.xlabel("data value")
    plt.ylabel("count")

    array_max = array.max()
    array_min = array.min()

    bins = np.bincount(array)

    points = [(x, bins[x]) for x in array]
    plt.xlim(array_min, array_max)

    for pt in points:
        plt.plot([pt[0], pt[0]], [0, pt[1]], 'grey')
    for k in truth_top_k_heavy_hitters:
        plt.text(k, bins[k], f"{k}", fontsize="x-small")

    plt.title(f"Frequency Distribution under {distribution_type}")
    plt.savefig(f"truth_frequency_{time.time_ns()}.png")


def plot_single_line(x_list: List[float], y_list: List[float], x_label: str, y_label: str, title: str, k) -> None:
    max_y = max(y_list)

    plt.plot(x_list, y_list)
    plt.plot(x_list, [max_y for _ in range(len(x_list))])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(x_list[-1], max_y, f"{y_label}={max_y}")

    plt.title(f"{title} (top-{k})")
    plt.savefig(f"{title}_{time.time_ns()}.png")


def plot_all_in_one(x_list: List[List], y_list: List[List], x_label: str, y_label: str, title: str, labels: List) -> None:

    plt.figure()
    for x, y, label in zip(x_list, y_list, labels):
        # print(x, y)
        plt.plot(x, y, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{title}_{time.time_ns()}.png")


def pr_N_mostFrequentNumber(arr, K=-1):

    mp = {}
    for i in range(len(arr)):
        if arr[i] in mp:
            mp[arr[i]] += 1
        else:
            mp[arr[i]] = 1

    a = sorted(mp.items(), key=lambda x: x[1],
               reverse=True)
    if K == -1:
        return [x[0] for x in a]

    a = a[:K]
    top_k = [x[0] for x in a]
    return top_k


def var(n, varepsilon, d: int):

    # print(self.varepsilon, d)
    p = exp(varepsilon) / (exp(varepsilon)+d-1)
    q = (1-p)/(d-1)
    # return p*log(p) + q*log(q)
    var_ = n*q*(1-q)/((p-q)**2)
    return var_


def binary_string(number, delta_s):
    binary = bin(number)[2:]
    append_0 = '0' * len(delta_s - bin(number)[2:])
    return append_0 + binary


def weight_score(n: int, varepsilon: float, d: int, batch: int) -> float:
    """_summary_

    Args:
        n (int): sampling client size
        varepsilon (float): privacy budget
        d (int): domain size of D

    Returns:
        float: weight
    """

    # weight_score_ = 1/var(n, varepsilon, d)
    p = exp(varepsilon) / (exp(varepsilon)+d-1)
    # q = (1-p)/(d-1)
    # print(n, p)
    weight_score_ = 20/(batch)
    # weight_score_ = 1
    print(f"batch:: {batch}, weight_score:", weight_score_)
    return weight_score_


def decode_result(self, result: List) -> List:
    for idx, number in enumerate(result):
        nbytes, rem = divmod(number.bit_length(), 8)
        if rem > 0:
            nbytes += 1
        result[idx] = int.to_bytes(
            number, byteorder="little", length=nbytes).decode(self.encoding)
    return result


def distance(x: int, y: int):
    diff = bin(x ^ y)
    dis = 0
    for i in diff:
        if i == '1':
            dis += 1
    return dis


def load_clients(filename="./dataset/triehh_clients.txt", k=sys.maxsize, restriction=-1, encode=True):
    with open(filename, 'rb') as f:
        clients = pickle.load(f)
    if restriction > 0:
        clients = random.sample(clients, restriction)
    truth_top_k = pr_N_mostFrequentNumber(clients, k)

    if not encode:
        return truth_top_k, clients

    for i in range(len(clients)):
        number = encode_word(clients[i])
        clients[i] = number
    for i in range(len(truth_top_k)):
        number = encode_word(truth_top_k[i])
        truth_top_k[i] = number

    return truth_top_k, clients
# Disable


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


def enablePrint():
    sys.stdout = sys.__stdout__


def plot(exp, save_filename="./results/zipfs.png", title="", line_type='.--', y_label="F1", x_label=r"$\varepsilon$", loc="upper left"):
    """_summary_

    Args:
        exp (_type_): {model_name: {x: [], y: []}}
        models (_type_): _description_
        save_filename (str, optional): _description_. Defaults to "./results/zipfs.png".
        y_label (str, optional): _description_. Defaults to "F1".
        x_label (regexp, optional): _description_. Defaults to r"$\varepsilon$".
    """
    models = list(exp.keys())
    color = {
        "PEM": 'teal', "PEM(GTU)": 'teal',
        "FedFT": "red", "FedFT(XTF)": 'red',
        'GTF': "yellowgreen",
        'XTU': "violet", "TrieHH": "lightslategrey",
        "OUE": "orange", "OLH": "blue", "THE": "green",
        "HR": "black", "CMS": "brown"
    }
    fig = plt.figure()
    for i in range(len(exp)):
        # if models[i] == "PEM":
        #     model_name = "PEM(GTU)"
        # elif models[i] == "XTF":
        #     model_name = "FedFT"
        # else:
        model_name = models[i]

        plt.plot(exp[models[i]]['x'], exp[models[i]]['y'],
                 line_type, color=color[model_name], label=models[i])

    plt.legend(loc=loc)
    plt.xlabel(x_label, fontdict={'fontsize': 16})
    plt.ylabel(y_label, fontdict={'fontsize': 16})
    # plt.style.use('seaborn-whitegrid')
    plt.title(title, fontdict={'fontsize': 16})
    if save_filename:
        plt.savefig(save_filename+".png")


def encode_words(word_list, return_binstring=False, binstring_width=64):
    encoded_words = []
    for i in range(len(word_list)):
        number = encode_word(word_list[i])
        if return_binstring:
            number = np.binary_repr(number, binstring_width)
        encoded_words.append(number)
    return encoded_words


def get_project_root() -> Path:
    return Path(__file__).parent


if __name__ == '__main__':
    xs = [[i for i in range(10)] for _ in range(2)]
    ys = [[2*i for i in range(10)] for _ in range(2)]
    labels = ["Tree", "Weight Tree"]

    plot_all_in_one(xs, ys, "varepsilon", "F1", "hello", labels)


def estimate_shrinkage_parameter(population_counts, overall_counts):
    n_populations = len(population_counts)
    B = np.sum((population_counts - overall_counts)**2) / (n_populations - 1)
    W = np.mean(overall_counts * (1 - overall_counts))
    shrinkage = (B - W) / B
    shrinkage = np.maximum(0, np.minimum(1, shrinkage))
    return shrinkage


def empirical_bayes_shrinkage(population_counts):
    overall_counts = np.mean(population_counts, axis=0)
    shrinkage = estimate_shrinkage_parameter(population_counts, overall_counts)
    return (1 - shrinkage) * overall_counts + shrinkage * population_counts


def get_top_frequent_words(populations_counts: list, top_N=2):
    population_counts = pd.DataFrame(populations_counts)

    empirical_bayes_counts = empirical_bayes_shrinkage(population_counts)
    avg_empirical_bayes_counts = np.mean(empirical_bayes_counts, axis=0)
    top_words_indices = avg_empirical_bayes_counts.argsort()[-top_N:][::-1]
    words = population_counts.columns
    top_words = [words[i] for i in top_words_indices]
    return top_words


def get_bayes_shrinkage_topk(top_N=2):
    DATA_PATH = "dataset/words_generate/"
    dict_ = []
    for n in range(2000, 10001, 1500):
        filename = f"words_generate_{n}_count"
        dict_i = load_words_count(f"{DATA_PATH}{filename}.txt")
        dict_.append(dict_i)

    top_words = get_top_frequent_words(dict_, top_N)

    return encode_words(top_words)
