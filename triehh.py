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

"""An implementation of Trie Heavy Hitters (TrieHH).

This is intended to implement and simulate the Trie Heavy Hitters (TrieHH)
protocol presented in Algorithm 1 of our AISTATS 2020 submission.
"""

import math
import os
import random
import numpy as np
from typing import List

from server import FAServerPEM
from utils import plot_single_line

np.random.seed(123499)

class ServerState(object):
    def __init__(self):
        self.quit_sign = False
        self.trie = {}


class SimulateTrieHH(FAServerPEM):
    """Simulation for TrieHH."""
    def __init__(self, n: int, m: int, k: int, varepsilon: float, iterations: int, round: int, clients: List = [], C_truth: List = [],\
            evaluate_type='F1', delta=2.3e-12, encoding="ascii", theta = None):
        super().__init__(n, m, k, varepsilon, iterations, round, clients, C_truth=C_truth, evaluate_type=evaluate_type)

        # super().__init__(self)
        
        self.trie_total_bits = 0
        self.msg_counts = self.m
        self.bit_len = math.ceil(self.m / self.iterations)
        print(f'Bit per round: {self.bit_len}')
        
        self.delta = delta
        # self.round = num_runs
        self.server_state = ServerState()
        self.encoding = encoding

        if not theta: 
            self._set_theta()
        else: self.theta = theta
        
        self._set_client_per_batch()

        
        if self.msg_counts < self.bit_len: # make true the first round goes.
            self.msg_counts = self.bit_len
        # else: self.msg_counts = msg_counts 


    def _set_theta(self):
        theta = 5  # initial guess
        delta_inverse = 1 / self.delta
        
        while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
            theta += 1
        while theta < np.e ** (self.varepsilon/(self.m // self.bit_len)) - 1:
            theta += 1
        self.theta = theta
        print(f'Theta used by TrieHH: {self.theta}')

    def _set_client_per_batch(self):
        # check Corollary 1 in our paper.
        # Done in _set_theta: We need to make sure theta >= np.e ** (self.epsilon/self.MAX_L) - 1
        self.clients_per_batch = int( self.n * (np.e ** (self.varepsilon / (self.m//self.bit_len)) - 1)/(self.theta * np.e ** (self.varepsilon/(self.m//self.bit_len)))) 

        print("Total client: ", self.n)
        print(f'Batch size used by TrieHH: {self.clients_per_batch}')

    def client_vote(self, number ):
        if self.trie_total_bits-self.bit_len == 0:  # trie initial state
            return 1
        # pre = number & ((1 << (self.trie_total_bits-self.bit_len))- 1) #TODO:
        pre = number >> (self.m - self.trie_total_bits + self.bit_len)
        if (pre not in self.server_state.trie.get(self.trie_total_bits-self.bit_len, {})):
            return 0
        return 1

    def client_updates(self):
        # I encourage you to think about how we could rewrite this function to do
        # one client update (i.e. return 1 vote from 1 chosen client).
        # Then you can have an outer for loop that iterates over chosen clients
        # and calls self.client_update() for each chosen and accumulates the votes.

        votes = {}
        voters = []
        for number in random.choices(self.clients, k=self.clients_per_batch):
            voters.append(number)

        for number in voters:
            vote_result = self.client_vote(number)

            pre = number >> (self.m-self.trie_total_bits) #TODO:
            votes[self.trie_total_bits] = votes.get(self.trie_total_bits, {})
            votes[self.trie_total_bits][pre] = votes[self.trie_total_bits].get(pre, 0) + vote_result
            # votes[pre] += vote_result
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
                self.server_state.trie[self.trie_total_bits] = self.server_state.trie.get(self.trie_total_bits, {})
                self.server_state.trie[self.trie_total_bits][prefix] = votes[prefix]
                self.server_state.quit_sign = False

    def predict_heavy_hitters(self):
        """Implementation of TrieHH."""
       
        self.server_state.trie.clear()
        self.trie_total_bits = 0
        
        self.trie_total_bits+= self.bit_len
        while self.trie_total_bits <= self.m and self.trie_total_bits <= self.msg_counts:
            votes = self.client_updates()
            self.server_update(votes[self.trie_total_bits])     
            if self.server_state.quit_sign:
                break
            self.trie_total_bits += self.bit_len
        if not self.server_state.trie:
            return []  # no hh found
        result_dict = self.server_state.trie[list(self.server_state.trie.keys())[-1]]
        result = sorted(result_dict.items(), key=lambda x: -x[1])
        
        return [hh[0] for hh in result[:self.k]]

    def decode_result(self, result: List) -> List:
        for idx, number in enumerate(result):
            nbytes, rem = divmod(number.bit_length(), 8)
            if rem > 0:
                nbytes += 1
            result[idx] = int.to_bytes(number, length=nbytes).decode(self.encoding)
        return result

    def server_run(self):
        """Simulate heavy hitters

        Returns:
            heavy_hitters: heavy hitter results
            prefix_size: the number of accurate bits of a number, remaining are 0s
        """

        evaluate_score = 0
        for rnd in range(self.round):
            result = self.predict_heavy_hitters()

            print(f"Truth ordering: {self.C_truth}")
            print(f"Predicted ordering: {result}")
            # print(result)
            evaluate_score += self.evaluate_module.evaluate(self.C_truth, result)

        evaluate_score /= self.round
        print(
            f"ROUND {rnd} :: varepsilon = {self.varepsilon}, {self.evaluate_type}= {evaluate_score:.2f}")
        return self.varepsilon, evaluate_score
        
    def server_run_plot_varepsilon(self, min_varepsilon, step_varepsilon, max_varepsilon):
        self.varepsilon = min_varepsilon
        varepsilon_list = []
        evaluate_score_list = []
        while self.varepsilon < max_varepsilon:


            varepsilon, evaluate_score = self.server_run()
            varepsilon_list.append(varepsilon)
            evaluate_score_list.append(evaluate_score)
            self.__change_varepsilon(self.varepsilon+step_varepsilon) 

        plot_single_line(varepsilon_list, evaluate_score_list, "varepsilon",
                         f"{self.evaluate_type}", f"{self.evaluate_type} vs varepsilon", k=self.k)
        return varepsilon_list, evaluate_score_list


    def __change_varepsilon(self, new_varepsilon):
        self.varepsilon = new_varepsilon

        self._set_theta()
        self._set_client_per_batch()

if __name__ == '__main__':
    n = 10000

    m = 32
    k = 8
    init_varepsilon = 2
    step_varepsilon = 0.3
    max_varepsilon = 12
    iterations =8 

    # sampling_rate = 1
    round = 2 
    delta = 1/(n**2)
    evaluate_module_type = "F1" # ["NDCG", "F1"]

    server = SimulateTrieHH(n, m, k, init_varepsilon, iterations, round, 
            delta=delta, evaluate_type=evaluate_module_type)
    # server.server_run()
    server.server_run_plot_varepsilon(
        init_varepsilon,  step_varepsilon, max_varepsilon)

    # visualize_frequency(server.clients, server.C_truth, server.client_distribution_type)
    