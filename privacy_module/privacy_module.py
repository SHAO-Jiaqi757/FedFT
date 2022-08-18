from math import exp
import random
import time
import numpy as np
from typing import Dict, List

from privacy_module.privacy_module_abc import PrivacyModuleABC
from utils import weight_score

random.seed(time.time_ns())
class PrivacyModule(PrivacyModuleABC):
    def __init__(self, varepsilon: float, D: Dict = {}, type: str = "GRR", bits_per_batch=-1, batch=-1):
        self.varepsilon = varepsilon
        self.D = D
        self.d = len(self.D) + 1
        self.type = type
        self.D_keys = sorted(list(D.keys()))
        self.bits_per_batch = bits_per_batch
        self.batch = batch

    def privacy_mechanism(self) -> callable:
        """_summary_
            Valid privacy mechanism types: ["None", "GRR", "OUE", "PreHashing"]
        Raises:
            Exception: Invalid privacy mechanism type

        Returns:
            callable: privacy mechanism with given type 
        """

        if self.type == "GRR" or self.type =="GRR_Weight":
            if not self.D:
                raise Exception("Dict is required for GRR")

            # print(self.varepsilon, d)
            p = exp(self.varepsilon) / (exp(self.varepsilon)+self.d-1)

            # print(f"Generate Random Response Probability: {p}")

            return self.__GRR(p)
        elif self.type == "None":
            return lambda x: x
        elif self.type == "OUE":
            return self.__OUE()
        elif self.type == "PreHashing":
            if self.bits_per_batch == -1 or self.batch == -1:
                raise Exception("`bits_per_batch` and `batch` are required for GRR")

            self.hashing_prefix = lambda v: (v & ((self.bits_per_batch*self.batch) << (self.bits_per_batch-1))) >> self.bits_per_batch

            return self.__PreHashing()


    def handle_response(self) -> callable:
        """_summary_
            Valid privacy mechanism types: ["None", "GRR", "OUE", "PreHashing"]
        Returns:
            callable: response handler with given type
        """
        if self.type == "GRR":
            return self.__handle_GRR_response()
        elif self.type == "None":
            return self.__handle_GRR_response()
        elif self.type == "OUE":
            return self.__handle_OUE_response()
        elif self.type == "PreHashing":
            return self.__handle_PreHashing_response()
        elif self.type == "GRR_Weight":
            return self.__handle_GRR_weight_response()
    
    def __handle_GRR_response(self):
        def __handle_GRR_response_(responses):
            for response in responses:
                if self.D.get(response, -1) != -1:
                    self.D[response] += 1
            return self.D

        return __handle_GRR_response_
    
    def __handle_GRR_weight_response(self):
        def __handle_GRR_weight_response_(responses):
            weight = weight_score(len(responses), self.varepsilon, self.d)
            for response in responses:
                self.D[response] = self.D.get(response, 0) + weight
            return self.D

        return __handle_GRR_weight_response_
    

    def __GRR(self, p: float):
        """_summary_

        Args:
            p (float): probability of replying truth answer
            d (int): domain size of D
        Returns: 
            GRR function with argument v
        """
        def GRR_(v: int):
           
            prob = random.random()

            if prob < p:
                return v
            else:
                random_choice_options = [item for item in self.D if item != v]
                random_choice_options.append(random.randint(0, 2**(self.batch*self.bits_per_batch)))
                return random.choice(random_choice_options)
        return GRR_


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


    def __handle_PreHashing_response(self) -> callable:
        
        def __handle_PreHashing_response_(responses):
            valid_responses = {}
            for response in responses:
                if response is not None:
                    if not self.D:
                        valid_responses[response] = valid_responses.get(response, 0) + 1
                    else:   
                        prefix_hash = self.hashing_prefix(response)

                        truth_part = (response & ((1 << self.bits_per_batch)-1))
                        
                        for decode_prefix_hash in self.D.get(prefix_hash, []):
                            response = (decode_prefix_hash << self.bits_per_batch)  + truth_part
                            valid_responses[response] = valid_responses.get(response, 0) + 1
            return valid_responses

        return __handle_PreHashing_response_ 

    def __PreHashing(self) -> callable:
        ...
        def __PreHashing_(v: int):
            if not self.D:  # initialize without hashing
                return v
            else: 

                prefix_hash = self.hashing_prefix(v)
                
                if self.D.get(prefix_hash, -1) != -1:
                    return (prefix_hash << self.bits_per_batch) + (v & ((1 << self.bits_per_batch)-1))
                else: 
                    return None

        return __PreHashing_


