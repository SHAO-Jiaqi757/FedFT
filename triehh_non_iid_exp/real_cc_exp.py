# coding=utf-8
# Copyright 2020 The Federated Heavy Hitters AISTATS 2020 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

# add absolute path of current path's parent to import modules
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

CUR_DIR_NAME = sys.path[0].split('/')[-1] 

import numpy as np
import pickle
from preprocess import *
from server_triehhServer import evaluate, SimulateTrieHH
from evaluate_module import EvaluateModule

DATA_PATH = "dataset/sentiment/"

def _get_non_iid_clusters_topk(k, word_count_file_list: list = []):
  """
  Return top k heavy hitters among non-iid clusters. 
  The rule of top k: tops [0] is the top k among the largest cluster, tops [1] is the top k among the second largest cluster, and so on.
  top_k is the top k among all clusters. e.g. top_k = [tops[0][0], tops[1][0], tops[2][0], tops[3][0], tops[4][0]] (k=4)
  Args:
    k: top k among clusters
    
  Return: 
    top_k: top k among all clusters
  """
  
  word_counts_main = {}

  for file_path_word_counts in word_count_file_list:
    word_counts = load_words_count(file_path_word_counts, k)
    counts_n = sum(word_counts.values())
    for key, v in word_counts.items():
      word_counts_main[key] = word_counts_main.get(key, 0) + v/counts_n

    sorted_word_counts = sorted(word_counts_main.items(), key=lambda x: x[1], reverse=True)
    top_k= [sorted_word_counts[i][0] for i in range(k)]

  return top_k

def get_non_iid_clusters_topk(k=5):
    
    word_count_file_list=[] 
    for month in ["sentiment", "reddit"]:
        filename = f"heavyhitters_{month}"
        word_count_file_list.append(f"{DATA_PATH}{filename}_count.txt")
        # cluster_top_k.append(cluster_truth_top_k)
        
    truth_hh = _get_non_iid_clusters_topk(k, word_count_file_list)
    
    return truth_hh


if __name__ == '__main__':

    # load dictionary
    # please provide your own dictionary if you would like to
    # run out-of-vocabulary experiments
    dictionary = 'dictionary.txt'
    # maximum word length
    max_word_len = 10
    max_k = 5 

    # length of longest word
    max_word_len = 10
    # delta for differential privacy
    delta = 2.3e-12
    # repeat simulation for num_runs times
    num_runs = 20
  
    filename = f"heavyhitters_combined"
    file_path = f"dataset/sentiment/{filename}.txt"
    client_path = generate_triehh_clients(file_path)
    truth_hh = get_non_iid_clusters_topk(max_k)
    print('truth_hh:', truth_hh)
    
    with open(client_path, 'rb') as fp:
        clients_top_word = pickle.load(fp)

    clients_top_word = np.array(clients_top_word)

    print('client count:', len(clients_top_word))

    f1_scores = []
    recall_scores = []
    # epsilon for differential privacy

    for epsilon in np.arange(0.5, 10, 1):
        simulate_triehh = SimulateTrieHH(client_path,
            max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)
        
        triehh_heavy_hitters = simulate_triehh.get_heavy_hitters()
        
        # print(f"top {max_k} words:", truth_hh)
        for evaluation_type in ["F1", "recall"]:

            evaluate_score, _ = evaluate(evaluation_type, triehh_heavy_hitters, truth_hh)
            if evaluation_type == "F1":
                f1_scores.append(evaluate_score)
            else: 
                recall_scores.append(evaluate_score)
