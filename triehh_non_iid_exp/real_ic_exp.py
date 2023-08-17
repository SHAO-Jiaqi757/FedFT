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
from utils import blockPrint, enablePrint

CUR_DIR_NAME = sys.path[0].split('/')[-1] 

import numpy as np
import pickle
from preprocess import *
from server_triehhServer import evaluate, SimulateTrieHH
from exp_generate_words import load_words_count



if __name__ == '__main__':

  # load dictionary
  # please provide your own dictionary if you would like to
  # run out-of-vocabulary experiments
  dictionary = 'dictionary.txt'
  # maximum word length
  max_word_len = 10
  
  blockPrint()
  for month in ["sentiment", "reddit"]: 
    filename = f"heavyhitters_{month}"
    file_path = f"dataset/sentiment/{filename}.txt"
    file_path_word_counts = f"dataset/sentiment/{filename}_count.txt" 
    client_path = generate_triehh_clients(file_path)
    
    # {'smog:': 244, 'pianist:': 103}
    word_counts = load_words_count(file_path_word_counts)
    
    with open(client_path, 'rb') as fp:
      clients_top_word = pickle.load(fp)
    
    # compute frequencies of top words
    top_word_frequencies = {}
    sum_num = sum(word_counts.values())
    for word in word_counts:
      top_word_frequencies[word] = word_counts[word] * 1.0 / sum_num

    clients_top_word = np.array(clients_top_word)
    
    # exp_triehh/clients_{file_name}.txt
    client_path_freqs = f"{CUR_DIR_NAME}/clients_freq_{filename}.txt"
    with open(client_path_freqs, 'wb') as fp:
      pickle.dump(top_word_frequencies, fp)

    # generate_sfp_clients(clients_top_word, max_word_len)

    print('client count:', len(clients_top_word))
    print('top word count:', len(word_counts))
    
    max_k = 5

    # length of longest word
    max_word_len = 10
    # delta for differential privacy
    delta = 2.3e-12
    # repeat simulation for num_runs times
    num_runs = 20
                
    f1_scores= []
    recall_scores = []
    # epsilon for differential privacy 0.5--9.5 with step 1
    
    for epsilon in np.arange(0.5, 10, 1):
        simulate_triehh = SimulateTrieHH(client_path,
            max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)
        
        triehh_heavy_hitters = simulate_triehh.get_heavy_hitters()
        
        
        truth_hh = list(word_counts.keys())[:max_k]

        
        # print(f"top {max_k} words:", truth_hh)
        for evaluation_type in ["F1", "recall"]:

            evaluate_score, _ = evaluate(evaluation_type, triehh_heavy_hitters, truth_hh)
            if evaluation_type == "F1":
                f1_scores.append(evaluate_score)
            else: 
                recall_scores.append(evaluate_score)
        