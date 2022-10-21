"""
GRR_X will be updated to GRRX
"""

from math import exp
import random
import time
import numpy as np
from typing import Dict, List

from privacy_module.privacy_module_abc import PrivacyModuleABC

random.seed(time.time_ns())
class PrivacyModule(PrivacyModuleABC):
    def __init__(self, varepsilon: float, D: Dict = {}, type: str = "GRR", s_i=0, required_bits = 0):
        self.varepsilon = varepsilon
        self.D = D
        self.type = type
        self.D_keys = sorted(list(D.keys()))
        self.s_i = s_i
        self.required_bits = required_bits
      

    def privacy_mechanism(self) -> callable:
        """_summary_
            Valid privacy mechanism types: ["None", "GRR", "OUE", "PreHashing"]
        Raises:
            Exception: Invalid privacy mechanism type

        Returns:
            callable: privacy mechanism with given type 
        """

        if self.type == "GRR":
            d = len(self.D)*2**self.required_bits
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d-1)

            # print(f"Generate Random Response Probability: {p}")
            return self.__GRR(p)
        elif self.type == "GRR_X":
            d = len(self.D) * 2**self.required_bits + 1
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d-1)
            return self.__GRR_X(p)
        elif self.type == "GRRX":
            d = len(self.D) * 2**self.required_bits + 1
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d-1)
            return self.__GRRX(p)

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
        elif self.type == "GRRX":
            return self.__handle_GRRX_response()
        elif self.type == "None":
            return self.__handle_GRR_response()
        elif self.type == "OUE":
            return self.__handle_OUE_response()
        else:
            print("Invaild Privacy Type")
    
    def __handle_GRR_response(self):
        def __handle_GRR_response_(responses):
            for response in responses:
                if response == None: continue
                
                self.D[response] = self.D.get(response, 0) + 1
            return self.D

        return __handle_GRR_response_

    def __handle_GRRX_response(self):
        def __handle_GRRX_response_(responses):
            C = {}
            for response in responses:
                if response == None: continue
                if response not in C:
                    C[response] = self.D.get(response[:self.s_i], 0) + 1 
                else:
                    C[response] += 1
            return C

        return __handle_GRRX_response_
     

    def __GRR(self, p: float):
        """_summary_

        Args:
            p (float): probability of replying truth answer
            d (int): domain size of D
        Returns: 
            GRR function with argument v
        """
        def GRR_(v: int):
            prefix_v = v >> self.required_bits
            suffix_v = v & ((1 << self.required_bits) - 1)
            if prefix_v not in self.D:
                return
            prob = random.random()

            if prob < p:
                return v
            else:
                random_choice_options = [i for i in range(2**self.required_bits)]
                random_choice_options.remove(suffix_v)
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
           
            prob = random.random()

            if prob < p:
                response = v    
            else:
                random_choice_options = list(self.D.keys())
                if v in random_choice_options:
                    X = random.randint(0, 2**(self.s_i))
                    random_choice_options.append(X) # randomly select X

                else:
                    random_choice_options.append(v) # X = v
                response = random.choice(random_choice_options)

            return response # random response
        return GRR_X

    def __GRRX(self, p: float):
        """_summary_

        Args:
            p (float): probability of replying truth answer
            d (int): domain size of D
        Returns: 
            GRR_X function with argument v
        """
        def GRRX(v: str):
           
            prob = random.random()

            if prob < p:
                response = v    
            else:
                random_choice_prefix_options = list(self.D.keys())
                random_append = np.binary_repr(random.randint(0, 2**self.required_bits-1), self.required_bits)
                random_choice_options = [ i + random_append for i in random_choice_prefix_options]

                if v[:self.s_i] in random_choice_prefix_options:
                
                    if self.s_i:
                        X = np.binary_repr(random.randint(0, 2**(self.s_i)-1), self.s_i) 
                    else:
                        X = ""
                    
                    X += random_append

                    random_choice_options.append(X) # randomly select X
        
                else:
                    random_choice_options.append(v) # X = v
                response = random.choice(random_choice_options)
            if len(response) != self.s_i + self.required_bits:
                print("warning")

            return response # random response
        return GRRX

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



