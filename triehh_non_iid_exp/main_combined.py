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

from main import SimulateTrieHH
# add absolute path of current path's parent to import modules
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

CUR_DIR_NAME = sys.path[0].split('/')[-1] 

import numpy as np
import pickle
from preprocess import *
from evaluate_module import EvaluateModule


def evaluate(evaluation_type:str, triehh_heavy_hitters: list, truth_hh: list):
  """_summary_

  Args:
      evaluation_type (str): F1, recall
      triehh_heavy_hitters (list): list of lists of heavy hitters for each run
      truth_hh (list):  list of lists of truth heavy hitters for each run
  Returns:
      float: mean of evaluation
      float: standard deviation of evaluation
  """
  evals = []
  evaluation = EvaluateModule(evaluation_type)
  for estimate_hh in triehh_heavy_hitters:
    eval = evaluation.evaluate(truth_top_k=truth_hh, estimate_top_k=estimate_hh)
    evals.append(eval)
  return np.mean(evals), np.std(evals)

if __name__ == '__main__':

    # load dictionary
    # please provide your own dictionary if you would like to
    # run out-of-vocabulary experiments
    dictionary = 'dictionary.txt'
    evaluate_type = "F1"
    # maximum word length
    max_word_len = 10
    max_k = 5 

    # length of longest word
    max_word_len = 10
    # delta for differential privacy
    delta = 2.3e-12
    # repeat simulation for num_runs times
    num_runs = 40
  
    filename = f"words_generate_combined"
    file_path = f"dataset/words_generate/{filename}.txt"
    client_path = generate_triehh_clients(file_path)
    truth_hh = _get_non_iid_clusters_topk(max_k)
    print('truth_hh:', truth_hh)
    
    with open(client_path, 'rb') as fp:
        clients_top_word = pickle.load(fp)

    clients_top_word = np.array(clients_top_word)

    print('client count:', len(clients_top_word))

    evals = []

    # epsilon for differential privacy
    for epsilon in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
        simulate_triehh = SimulateTrieHH(client_path,
        max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)

        triehh_heavy_hitters = simulate_triehh.get_heavy_hitters()

        evaluate_score, _ = evaluate(evaluate_type, triehh_heavy_hitters, truth_hh)
        evals.append(evaluate_score)

        
        # print -->  a|b ... | ... for a, b in evals, a, b
        print(f"TrieHH, {evaluate_type}, {','.join([str(round(a, 3)) for a in evals])}|")
        
            
