# %%
import random
from sympy import isprime
import math
import numpy as np


class OHE_2RR:
    def __init__(self, D_t, eps, seed=random.seed()):
        self.p1 = 1.0 / (math.exp(eps) + 1)
        self.p2 = 0.5
        self.K= len(D_t) # domain size
        self.D = D_t
        self.eps = eps
       
        self.mapping_D_dict = {d: i for i, d in enumerate(D_t)}
        self.mapping_D = [_ for _ in range(self.K)]

    def local_randomizer(self, x):
        map_x = self.mapping_D_dict.get(x, -1)
        if map_x== -1:
            return 
        else: x = map_x
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

    def estimate_all_freqs(self, messages):
        
        
        ret = [0] * self.K
        message_count = len(messages)
        if message_count == 0: return ret, self.D
        exp_eps = math.exp(self.eps)

        messages = np.array(messages)
        sum_d = np.sum(messages, axis=0)
        for i in range(self.K):
            ret[i] = 2 * (1 + exp_eps) / (exp_eps - 1) * sum_d[i] - message_count / (exp_eps - 1)
        
        pairs = [(ret[d], self.D[d]) for d in self.mapping_D]
        pairs.sort(reverse=True)
        
        sorted_freqs, sorted_items = zip(*pairs)
        
        return sorted_freqs, sorted_items
