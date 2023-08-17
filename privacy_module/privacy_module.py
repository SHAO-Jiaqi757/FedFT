from math import exp
import random
import time
import numpy as np
from typing import Dict, List
import hashlib
from sympy import isprime

from privacy_module.privacy_module_abc import PrivacyModuleABC

# random.seed(time.time_ns())
class PrivacyModule(PrivacyModuleABC):
    def __init__(self, varepsilon: float, D: Dict = {}, type: str = "GRR", s_i=0, required_bits = 0, client_id=-1):
        assert type in ["GRR", "GRR_X", "OHE_2RR", "PIRAPPOR", "None", "OUE"], "Invalid privacy mechanism type"
        self.varepsilon = varepsilon
        self.D = D
        self.type = type
        self.D_keys = sorted(list(D.keys()))
        self.s_i = s_i
        self.required_bits = required_bits
        self.client_id = client_id


        if self.type == "PIRAPPOR" or self.type =="OHE_2RR":
            self.D_extend = []
            for d in self.D:
                self.D_extend += [ (d << self.required_bits) + i for i in range(2**self.required_bits)]
            self.mapping_D_dict = {d: i for i, d in enumerate(self.D_extend)}
            self.mapping_D = [_ for _ in range(len(self.D_extend))]
            self.K = len(self.D_extend)
            
        if self.type == "PIRAPPOR":
            
            self.p2 = 0.5
            self.q = int((1 + self.p2 * (exp(varepsilon) - 1)) / (1.0 - self.p2))
            if self.q % 2 == 0 and self.q != 2:
                self.q -= 1
            while not isprime(self.q): 
                self.q -= 2
            self.zero_threshold = self.q - 1
            # self.p1 = 1 - 1.0 * self.zero_threshold / self.q
            self.p1 = 1.0 / (exp(varepsilon) + 1)
            self.t = 0
            self.qpows = [1]
            while self.qpows[-1] <= self.K:
                self.t += 1
                self.qpows.append(self.qpows[-1] * self.q)
            self.K = self.qpows[-1] - 1
            self.alpha = 1.0 / (1.0 - self.p1 - self.p2 + 1e-3)
            self.beta = -self.p1 / (1.0 - self.p1 - self.p2 + 1e-3)
            self.unif_intq = lambda: random.randint(0, self.q - 1)
            self.unif_int0 = lambda: random.randint(0, self.zero_threshold - 1)
            self.unif_int1 = lambda: random.randint(self.zero_threshold, self.q - 1)


    def privacy_mechanism(self) -> callable:
        """_summary_
        Raises:
            Exception: Invalid privacy mechanism type

        Returns:
            callable: privacy mechanism with given type 
        """

        if self.type == "GRR":
            d = len(self.D)*2**self.required_bits
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d-1)
            self.p = p
            # print(f"Generate Random Response Probability: {p}")
            return self.__GRR(p)
        elif self.type == "GRR_X":
            d = len(self.D) * 2**self.required_bits + 1
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d-1)
       
            self.p = p
            return self.__GRR_X(p)
        elif self.type == "OHE_2RR":
            return self.__OHE_2RR()
        elif self.type == "PIRAPPOR":
            return self.__PIRAPPOR()
        
        elif self.type == "None":
            return lambda x: x
        elif self.type == "OUE":
            return self.__OUE()
        else:
            print("Invaild Privacy Type")
            
    
        
        
    def handle_response(self) -> callable:
        """_summary_
            Valid privacy mechanism types: ["None", "GRR", "OUE", "PreHashing"]
        Returns:
            callable: response handler with given type
        """
        if self.type == "GRR" or self.type == "GRR_X":
            return self.__handle_GRR_response()
        elif self.type == "None":
            return self.__handle_GRR_response()
        elif self.type == "PIRAPPOR":
            return self.__handle_PIRAPPOR_response()
        elif self.type == "OHE_2RR":
            return self.__handle_OHE2RR_response()
        elif self.type == "OUE":
            return self.__handle_OUE_response()
        else:
            print("Invaild Privacy Type")
    
    def __handle_GRR_response(self):
        def __handle_GRR_response_(responses):
            R = {}
            for response in responses:
                if response == None: continue
                
                R[response] = R.get(response, 0) + 1
            return R

        return __handle_GRR_response_
    def __handle_OHE2RR_response(self):
        
        def estimate_all_freqs(messages):
            ret = [0] * self.K
            message_count = len(messages)
            if message_count == 0:
                return {item: 0 for item in self.D_extend}
            exp_eps = exp(self.varepsilon)
            messages = np.array(messages)
            sum_d = np.sum(messages, axis=0)
            
            for i in range(self.K):
                ret[i] = 2 * (1 + exp_eps) / (exp_eps - 1) * sum_d[i] - message_count / (exp_eps - 1)
            
            pairs = [(ret[d], d) for d in self.mapping_D]
            pairs.sort(reverse=True)
                        
            return {self.D_extend[item]: freq for freq, item in pairs}
        return estimate_all_freqs
    def __handle_PIRAPPOR_response(self):
        
        def estimate_freq(x, messages):
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

        def dp_bottom_up(y):
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

        def estimate_all_freqs(messages):
            ret = [0] * self.K
            message_count = len(messages)
            
            if message_count < self.q * self.q * self.q:
                for x in range(self.K):
                    ret[x] = estimate_freq(x, messages)
            else:
                y = [0] * (self.qpows[self.t] * self.q)
                for m in messages:
                    y[m] += 1
                T = dp_bottom_up(y)
                alpha_T = [self.alpha * freq for freq in T]
                beta_message_count = self.beta * message_count
                for x in range(self.K):
                    ret[x] = alpha_T[x] + beta_message_count
                
            pairs = [(ret[d], d) for d in self.mapping_D]
            pairs.sort(reverse=True)
            return {self.D_extend[item]: freq for freq, item in pairs}
        
        return estimate_all_freqs
     
    def __handle_OUE_response(self) -> callable:

        """
        Handle Optimized Unary Encoding response function
        """
        def __handle_OUE_response_(responses: List[List[int]]):
            """_summary_

            Args:
                responses (List[List[int]]): All clients' responses, where each client replays the unary encoded response

            Returns:
                _type_: results of Aggregating clients' Optimized Unary Encoding responses
            """
            response_aggregate = np.sum(responses, axis=0) 
            for index, count in enumerate(response_aggregate):
                key = self.D_keys[index]
                self.D[key] += count
            return self.D
        return __handle_OUE_response_



    def __GRR(self, p: float):
        """_summary_

        Args:
            p (float): probability of replying truth answer
            d (int): domain size of D
        Returns: 
            GRR function with argument v
        """
        def GRR_(v: int):
            prefix_v = v >> self.required_bits # v with self.s_i bits, prefix_v is the prefix matching for the parent nodes.
            
            if prefix_v not in self.D: # v not in candidate domain, return None.
                return

            prob = random.random()
            if prob < p:
                return v
            else:
                random_choice_options = []
                for prefix in self.D:
                    for i in range(2**self.required_bits):
                        y = (prefix << self.required_bits) + i
                        if y == v: continue
                        random_choice_options.append(y)

                return random.choice(random_choice_options) # random response
        return GRR_

    def __GRR_X(self, p: float):
        """_summary_

        Args:
            p (float): probability of replying truth answer
            d (int): domain size of D
        Returns: 
            GRR_X function with argument v
        """
        def GRR_X(v: int):
            # prefix_v = v >> self.required_bits
            prob = random.random()

            if prob < p:
                return v   # truely response 
            else:
                # local response domain
                random_choice_options = []
                for prefix in self.D:
                    for i in range(2**self.required_bits):
                        y = (prefix << self.required_bits) + i
                        if v == y: # v in candidiate domain 
                            # if prefix_v in self.D:
                            y = client_hash(v, self.client_id, self.s_i)
                        random_choice_options.append(y)
                        
                return random.choice(random_choice_options) # random response

    
        return GRR_X

    def __OHE_2RR(self):
        p1 = 1.0 / (exp(self.varepsilon) + 1)
        p2 = 0.5

        def local_randomizer(x):
            map_x = self.mapping_D_dict.get(x, -1)
            if map_x == -1:
                return 
            else:
                x= map_x
            ret = [0] * self.K
            # bernoulli p = 1/2
            for i in range(self.K):
                p = random.random()
                if i == x:
                    if p < p2:
                        ret[i] = 1
                    else: ret[i] = 0
                else:
                    if p < p1:
                        ret[i] = 1
                    else:
                        ret[i] = 0
            return ret
        return local_randomizer
    
    def __OUE(self):
        """
        Optimized Unary Encoding response function
        """
        def __OUE_(v) -> List:
            """_summary_

            Args:
                v (_type_): input value

            Returns:
                List: Optimized Unary Encoding response
            """
            response = []
            for key in self.D_keys:
                if key == v:
                    p = random.random()
                    response.append(1 if p < 1/2 else 0)
                    
                else:
                    p = random.random()
                    response.append(1 if p < 1/(exp(self.varepsilon) + 1) else 0)
            return response

        return __OUE_

    def __PIRAPPOR(self):
        def local_randomizer(x):
            map_x = self.mapping_D_dict.get(x, -1)
            if map_x == -1:
                return
            else: x= map_x
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
        return local_randomizer
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

def client_hash(v, client_id, s):
    # Convert the client id and input v to bytes, if they're not already
    if not isinstance(v, bytes):
        v = str(v).encode()
    if not isinstance(client_id, bytes):
        client_id = str(client_id).encode()

    # Create a sha256 hash object
    hasher = hashlib.sha256()

    # Update the hash with the input v
    hasher.update(v)

    # Get the digest
    v_hash = hasher.digest()

    # Create a new hasher for the client_id
    client_hasher = hashlib.sha256()
    client_hasher.update(client_id)
    client_id_hash = client_hasher.digest()

    # XOR the bytes of the hashes together
    combined_hash = bytes([v ^ c for v, c in zip(v_hash, client_id_hash)])

    # Truncate or pad the result to get a hash of exactly s bits
    if s < 8*len(combined_hash):
        # If the result is too long, truncate it
        combined_hash = combined_hash[:s//8]
    elif s > 8*len(combined_hash):
        # If the result is too short, pad it with zeros
        combined_hash += b'\0' * ((s - 8*len(combined_hash) + 7) // 8)

    # Convert the result to an integer in the range [0, 2**s)
    y = int.from_bytes(combined_hash, byteorder='big')

    # Return the resulting hash
    return y
