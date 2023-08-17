import sys
# add absolute path of current path's parent to import modules
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

CUR_DIR_NAME = sys.path[0].split('/')[-1] 


import numpy as np
import pickle
from collections import OrderedDict, defaultdict
from preprocess import *
import math
import random
from exp_generate_words import load_words_count
from evaluate_module import EvaluateModule




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
        self.server_state.trie[prefix] = votes[prefix]
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
      raw_result = sorted(self.server_state.trie.items(), key=lambda x: x[1], reverse=True)
      results = []
      for word, _ in raw_result:
        if word[-1:] == '$':
          results.append(word.rstrip('$'))
      # print(f'Discovered {len(results)} heavy hitters in run #{run+1}')
      # print(results)
      heavy_hitters.append(results)
    return heavy_hitters

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
    if len(estimate_hh) < len(truth_hh):
        estimate_hh += [''] * (len(truth_hh) - len(estimate_hh))
    else:
        estimate_hh = estimate_hh[:len(truth_hh)]
    eval = evaluation.evaluate(truth_top_k=truth_hh, estimate_top_k=estimate_hh)
    evals.append(eval)
  return np.mean(evals), np.std(evals)
