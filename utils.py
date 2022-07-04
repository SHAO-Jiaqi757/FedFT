from math import log
import sys
import random
import time
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np

random.seed(time.time_ns())

def sort_by_frequency(array, k=sys.maxsize) -> list:
    return np.argsort(-np.bincount(array))[:k]  

def visualize_frequency(array, truth_top_k_heavy_hitters, k=sys.maxsize):
    plt.figure()
    plt.xlabel("data value")
    plt.ylabel("count")
    bin, _, _ = plt.hist(array[:k], bins=len(np.bincount(array)), rwidth=0.9)
    for k in truth_top_k_heavy_hitters:
        plt.text(k, bin[k], f"{k}", fontsize="x-small")
    plt.title("Frequency Distribution")
    plt.savefig(f"truth_frequency_{time.time_ns()}.png")

def plot_single_line(x_list: List[float], y_list: List[float], x_label: str, y_label: str, title: str, k) -> None:
    max_y = max(y_list)

    plt.plot(x_list, y_list)
    plt.plot(x_list, [max_y for _ in range(len(x_list))])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(x_list[-1],max_y, f"ndcg={max_y}")

    plt.title(f"{title} (top-{k})")
    plt.savefig(f"{title}_{time.time_ns()}.png")

def plot_all_in_one(x_list: List[List], y_list: List[List], x_label: str, y_label: str, title: str) -> None:
    for x, y in zip(x_list, y_list):
        # print(x, y)
        plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{title}_{time.time_ns()}.png")
    



