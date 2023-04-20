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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy
import scipy.stats
from collections import OrderedDict, defaultdict
from preprocess import *
import math
import random
from exp_generate_words import load_words_count
from evaluate_module import EvaluateModule

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


"""An implementation of Trie Heavy Hitters (TrieHH).

This is intended to implement and simulate the Trie Heavy Hitters (TrieHH)
protocol presented in Algorithm 1 of our AISTATS 2020 submission.
"""



class ServerState(object):
  def __init__(self):
    self.quit_sign = False
    self.trie = {}

class SimulateTrieHH(object):
  """Simulation for TrieHH."""

  def __init__(self, client_path, max_word_len=10, epsilon=1.0, delta = 2.3e-12, num_runs=5):
    self.MAX_L = max_word_len
    self.delta = delta
    self.epsilon = epsilon
    self.num_runs = num_runs
    self.client_path =client_path
    self.clients = []

    self.client_num = 0
    self.server_state = ServerState()
    self._init_clients()
    self._set_theta()
    self._set_batch_size()
    
  def set_epsilon(self, epsilon):
    self.epsilon = epsilon
    self._set_theta()
    self._set_batch_size()

  def _init_clients(self):
    """Initialization of the dictionary."""
    with open(self.client_path, 'rb') as fp:
      self.clients = pickle.load(fp)
    self.client_num = len(self.clients)
    print(f'Total number of clients: {self.client_num}')

  def _set_theta(self):
    theta = 5  # initial guess
    delta_inverse = 1 / self.delta
    while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
      theta += 1
    while theta < np.e ** (self.epsilon/self.MAX_L) - 1:
      theta += 1
    self.theta = theta
    print(f'Theta used by TrieHH: {self.theta}')

  def _set_batch_size(self):
    # check Corollary 1 in our paper.
    # Done in _set_theta: We need to make sure theta >= np.e ** (self.epsilon/self.MAX_L) - 1
    self.batch_size = int( self.client_num * (np.e ** (self.epsilon/self.MAX_L) - 1)/(self.theta * np.e ** (self.epsilon/self.MAX_L)))
    print(f'Batch size used by TrieHH: {self.batch_size}')

  def client_vote(self, word, r):
    if len(word) < r:
      return 0

    pre = word[0:r-1]
    if pre and (pre not in self.server_state.trie):
      return 0

    return 1

  def client_updates(self, r):
    # I encourage you to think about how we could rewrite this function to do
    # one client update (i.e. return 1 vote from 1 chosen client).
    # Then you can have an outer for loop that iterates over chosen clients
    # and calls self.client_update() for each chosen and accumulates the votes.

    votes = defaultdict(int)
    voters = []
    for word in random.sample(self.clients, self.batch_size):
      voters.append(word)

    for word in voters:
      vote_result = self.client_vote(word, r)
      if vote_result > 0:
        votes[word[0:r]] += vote_result
    return votes

  def server_update(self, votes):
    # It might make more sense to define a small class called server_state
    # server_state can track 2 things: 1) updated trie, and 2) quit_sign
    # server_state can be initialized in the constructor of SimulateTrieHH
    # and server_update would just update server_state
    # (i.e, it would update self.server_state.trie & self.server_state.quit_sign)
    self.server_state.quit_sign = True
    for prefix in votes:
      if votes[prefix] >= self.theta:
        self.server_state.trie[prefix] = None
        self.server_state.quit_sign = False

  def start(self, batch_size):
    """Implementation of TrieHH."""
    self.server_state.trie.clear()
    r = 1
    while True:
      votes = self.client_updates(r)
      self.server_update(votes)
      r += 1
      if self.server_state.quit_sign or r > self.MAX_L:
        break

  def get_heavy_hitters(self):
    heavy_hitters = []
    for run in range(self.num_runs):
      self.start(self.batch_size)
      raw_result = self.server_state.trie.keys()
      results = []
      for word in raw_result:
        if word[-1:] == '$':
          results.append(word.rstrip('$'))
      # print(f'Discovered {len(results)} heavy hitters in run #{run+1}')
      # print(results)
      heavy_hitters.append(results)
    return heavy_hitters


  



class Plot(object):

  def __init__(self, max_k, client_freq_path):
    self.confidence = .95
    self.max_k = max_k
    self.client_freq_path = client_freq_path
    self._load_true_frequencies()

  def _load_true_frequencies(self):
    """Initialization of the dictionary."""
    with open(self.client_freq_path, 'rb') as fp:
      self.true_frequencies = pickle.load(fp)

  def get_mean_u_l(self, recall_values):
    data_mean = []
    ub = []
    lb = []
    for K in range(10, self.max_k):
      curr_mean = np.mean(recall_values[K])
      data_mean.append(curr_mean)
      n = len(recall_values[K])
      std_err = scipy.stats.sem(recall_values[K])
      h = std_err * scipy.stats.t.ppf((1 + self.confidence) / 2, n - 1)
      lb.append(curr_mean - h)
      ub.append(curr_mean + h)
    mean_u_l = [data_mean, ub, lb]
    return mean_u_l

  def precision(self, result):
    all_words_key = self.true_frequencies.keys()
    precision = 0
    for word in result:
      if word in all_words_key:
        precision += 1
    precision /= len(result)
    return precision

  def plot_f1_scores(self, triehh_all_results, sfp_all_results, epsilon):
    # CHANGE "apple" TO "sfp"
    # CLEAN THIS (REMOVE ANY EXCESS CODE NOT USED ANYMORE

    sorted_all = OrderedDict(sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse = True))
    top_words = list(sorted_all.keys())[:self.max_k]

    all_f1_triehh = []
    all_f1_sfp = []
    k_values = []

    for K in range(10, self.max_k):
      k_values.append(K)

    f1_values_triehh = {}
    f1_values_sfp = {}
    f1_values_inter = {}

    for K in range(10, self.max_k):
      f1_values_triehh[K] = []
      f1_values_sfp[K] = []
      f1_values_inter[K] = []

    for triehh_result in triehh_all_results:
      for K in range(10, self.max_k):
        recall = 0
        for i in range(K):
          if top_words[i] in triehh_result:
            recall += 1
        recall = recall * 1.0/K
        f1_values_triehh[K].append(2*recall/(recall + 1))
    all_f1_triehh = self.get_mean_u_l(f1_values_triehh)

    sfp_precision_list = []
    for sfp_result in sfp_all_results:
      precision_sfp = self.precision(sfp_result)
      sfp_precision_list.append(precision_sfp)
      for K in range(10, self.max_k):
        recall_sfp = 0
        for i in range(K):
          if top_words[i] in sfp_result:
            recall_sfp += 1
        recall_sfp = recall_sfp * 1.0/K
        f1_values_sfp[K].append(2*precision_sfp*recall_sfp/(precision_sfp + recall_sfp))
    all_f1_sfp = self.get_mean_u_l(f1_values_sfp)

    _, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xlabel('K', fontsize=16)
    ax1.set_ylabel('F1 Score', fontsize=16)


    ax1.plot(k_values, all_f1_triehh[0], color = 'purple', alpha = 1, label=r'TrieHH, $\varepsilon$ = '+str(epsilon))
    ax1.fill_between(k_values, all_f1_triehh[2], all_f1_triehh[1], color = 'violet', alpha = 0.3)

    ax1.plot(k_values, all_f1_sfp[0], color = 'blue', alpha = 1, label=r'SFP, $\varepsilon$ = '+str(epsilon))
    ax1.fill_between(k_values, all_f1_sfp[2], all_f1_sfp[1], color = 'skyblue', alpha = 0.3)


    plt.legend(loc=4, fontsize=14)

    plt.title('Top K F1 Score vs. K (Single Word)', fontsize=14)
    plt.savefig("f1_single.eps")
    plt.savefig("f1_single.png",  bbox_inches="tight")
    plt.close()

def main():

  # length of longest word
  max_word_len = 10
  # epsilon for differential privacy
  epsilon = 2 
  # delta for differential privacy
  delta = 2.3e-12

  # repeat simulation for num_runs times
  num_runs = 20
  
  simulate_triehh = SimulateTrieHH(
      max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)
  triehh_heavy_hitters = simulate_triehh.get_heavy_hitters()

  # simulate_sfp = SimulateSFP(
  #     max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)
  # sfp_heavy_hitters = simulate_sfp.get_heavy_hitters()

  # plot = Plot(max_k)
  # plot.plot_f1_scores(triehh_heavy_hitters, sfp_heavy_hitters, epsilon)




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
    evaluate_type = "recall"
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
    truth_hh = get_non_iid_clusters_topk(max_k)
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
        print(f"Combined|{'|'.join([str(round(a, 3)) for a in evals])}|")
        
            
