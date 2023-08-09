# %%
import random
from sympy import isprime
import math



class PIRAPPOR:
    def __init__(self, K_, eps, seed=random.seed()):
        self.p1 = 1.0 / (math.exp(eps) + 1)
        self.p2 = 0.5
        self.q = int((1 + self.p2 * (math.exp(eps) - 1)) / (1.0 - self.p2))
        if self.q % 2 == 0 and self.q != 2:
            self.q -= 1
        while not isprime(self.q): 
            self.q -= 2
        self.zero_threshold = self.q - 1
        self.p1 = 1 - 1.0 * self.zero_threshold / self.q
        self.t = 0
        self.qpows = [1]
        while self.qpows[-1] <= K_:
            self.t += 1
            self.qpows.append(self.qpows[-1] * self.q)
        self.K = self.qpows[-1] - 1
        self.alpha = 1.0 / (1.0 - self.p1 - self.p2)
        self.beta = -self.p1 / (1.0 - self.p1 - self.p2)
        self.unif_intq = lambda: random.randint(0, self.q - 1)
        self.unif_int0 = lambda: random.randint(0, self.zero_threshold - 1)
        self.unif_int1 = lambda: random.randint(self.zero_threshold, self.q - 1)
    
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
        ret = [0] * (self.t + 1)
        v = self.num_to_vec(x + 1, self.t)
        dot_prod = 0
        for i in range(self.t):
            ret[i] = self.unif_intq()
            dot_prod = (dot_prod + ret[i] * v[i]) % self.q
        if random.random() <= self.p2:
            ret[self.t] = (self.q + self.unif_int0() - dot_prod) % self.q
        else:
            ret[self.t] = (self.q + self.unif_int1() - dot_prod) % self.q
        return self.vec_to_num(ret)

    def estimate_freq(self, x, messages):
        v = self.num_to_vec(x + 1, self.t)
        cnt = 0
        for m in messages:
            dot_prod = 0
            u = self.num_to_vec(m, self.t + 1)
            for i in range(len(v)):
                dot_prod = (dot_prod + u[i] * v[i]) % self.q
            dot_prod = (dot_prod + u[self.t]) % self.q
            if dot_prod >= self.zero_threshold:
                cnt += 1
        return self.alpha * cnt + self.beta * len(messages)

    def dp_bottom_up(self, y):
        N = self.qpows[self.t] * self.q
        last = [0] * N
        next_ = [0] * N

        for a in range(self.qpows[self.t]):
            for z in range(self.q):
                last[a * self.q + z] = y[a * self.q + z]

        prevA = self.qpows[self.t]
        prevB = 1
        for length in range(self.t - 1, -1, -1):
            next_ = [0] * N
            curA = self.qpows[length]
            curB = self.qpows[self.t - length]
            for b in range(curB):
                val = b % self.qpows[self.t - length - 1]
                first_b_digit = (b - val) // self.qpows[self.t - length - 1]
                for a in range(curA):
                    for z in range(self.q):
                        for d in range(self.q):
                            next_[b * curA * self.q + a * self.q + z] += last[val * prevA * self.q + (a * self.q + d) * self.q + (self.q * d + z - first_b_digit * d) % self.q]
            last, next_ = next_, last
            prevA = curA
            prevB = curB
        
        ret = [0] * self.K
        for i in range(1, self.K + 1):
            for z in range(self.zero_threshold, self.q):
                ret[i - 1] += last[i * self.q + z]
        return ret

    def estimate_all_freqs(self, messages, D):
        ret = [0] * self.K
        message_count = len(messages)
        
        if message_count < self.q * self.q * self.q:
            for x in range(self.K):
                ret[x] = self.estimate_freq(x, messages)
        else:
            y = [0] * (self.qpows[self.t] * self.q)
            for m in messages:
                y[m] += 1
            T = self.dp_bottom_up(y)
            alpha_T = [self.alpha * freq for freq in T]
            beta_message_count = self.beta * message_count
            for x in range(self.K):
                ret[x] = alpha_T[x] + beta_message_count
            
        pairs = [(ret[d], d) for d in D]
        pairs.sort(reverse=True)
        sorted_freqs, sorted_items = zip(*pairs)
        return sorted_freqs, sorted_items

