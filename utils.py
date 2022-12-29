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

random.seed(time.time_ns())

def sort_by_frequency(array, k=sys.maxsize) -> list:
    return np.argsort(-np.bincount(array))[:k]  

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
    plt.text(x_list[-1],max_y, f"{y_label}={max_y}")

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
    if K==-1: return [x[0] for x in a]
    
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
    binary =  bin(number)[2:]
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
        result[idx] = int.to_bytes(number, byteorder="little", length=nbytes).decode(self.encoding)
    return result


def distance(x: int, y: int): 
    diff = bin(x^y)
    dis = 0
    for i in diff: 
        if i == '1': 
            dis += 1
    return dis
    
def load_clients(filename="./dataset/triehh_clients.txt", k = sys.maxsize, restriction=-1, encode=True):
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


def plot(exp, save_filename = "./results/zipfs.png",title="", line_type = '.--', y_label="F1", x_label=r"$\varepsilon$", loc="upper left"):
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
        'GTF':"yellowgreen",
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
    
        plt.plot(exp[models[i]]['x'], exp[models[i]]['y'], line_type, color=color[model_name], label=models[i])

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
