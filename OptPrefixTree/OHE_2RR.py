# %%
import random
from sympy import isprime
import math
import numpy as np


class OHE_2RR:
    def __init__(self, K_, eps, seed=random.seed()):
        self.p1 = 1.0 / (math.exp(eps) + 1)
        self.p2 = 0.5
        self.K= K_ # Domain size
        self.eps = eps
    
    def num_to_vec(self, x, l):
        ret = [0] * l
        for i in range(l):
            ret[l - 1 - i] = x % self.q
            x //= self.q
        return ret

    def vec_to_num(self, v):
        x = 0
        qpow = 1
        for i in range(len(v)):
            x += v[len(v) - 1 - i] * qpow
            qpow *= self.q
        return x

    def local_randomizer(self, x):
        ret = [0] * self.K
        # bernoulli p = 1/2
        for i in range(self.K):
            p = random.random()
            if i == x:
                if p < self.p2:
                    ret[i] = 1
                else: ret[i] = 0
            else:
                if p < self.p1:
                    ret[i] = 1
                else:
                    ret[i] = 0
        return ret

    def estimate_all_freqs(self, messages, D):
        ret = [0] * self.K
        message_count = len(messages)
        exp_eps = math.exp(self.eps)
        
        messages = np.array(messages)
        sum_d = np.sum(messages, axis=0)
        for i in range(self.K):
            ret[i] = 2 * (1 + exp_eps) / (exp_eps - 1) * sum_d[i] - message_count / (exp_eps - 1)
        
        pairs = [(ret[d], d) for d in D]
        pairs.sort(reverse=True)
        sorted_freqs, sorted_items = zip(*pairs)
        
        return sorted_freqs, sorted_items
