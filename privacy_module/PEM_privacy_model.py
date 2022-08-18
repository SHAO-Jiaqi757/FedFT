from math import exp
import random
import time
import numpy as np
from typing import Dict, List

from privacy_module.privacy_module_abc import PrivacyModuleABC
from utils import weight_score

class PEMPrivacyModule(PrivacyModuleABC):
    def __init__(self, varepsilon: float, D: Dict = {}, type: str = "GRR", bits_per_batch=-1, batch=-1):
        self.varepsilon = varepsilon
        self.D = D
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
  
                # self.D = {0:0, 1:0}

            d = len(self.D)
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d)

            # print(f"Generate Random Response Probability: {p}")

            return self.__GRR(p)
        elif self.type == "None":
            return lambda x: x
        elif self.type == "OUE":
            return self.__OUE()
        else:
            raise Exception("Invalid privacy mechanism type")

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
        else:
            raise Exception("Invalid privacy mechanism type")
            
    def __handle_GRR_response(self):
        def __handle_GRR_response_(responses):
            for response in responses:
                if self.D.get(response, -1) != -1:
                    self.D[response] += 1
            return self.D

        return __handle_GRR_response_
    
    def __GRR(self, p: float):
        """_summary_

        Args:
            p (float): probability of replying truth answer
            d (int): domain size of D
        Returns: 
            GRR function with argument v
        """
        def GRR_(v: int):
            if v not in self.D:
                print(f"{v} not in D")
                return

            prob = random.random()

            if prob < p:
                return v
            else:
                random_choice_options = [item for item in self.D if item != v]
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
