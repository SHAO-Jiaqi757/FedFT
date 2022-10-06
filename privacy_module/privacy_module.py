from math import exp
import random
import time
import numpy as np
from typing import Dict, List

from privacy_module.privacy_module_abc import PrivacyModuleABC
from utils import weight_score

random.seed(time.time_ns())
class PrivacyModule(PrivacyModuleABC):
    def __init__(self, varepsilon: float, D: Dict = {}, type: str = "GRR", s_i=0, batch=-1):
        self.varepsilon = varepsilon
        self.D = D
        self.d = len(self.D)
        self.type = type
        self.D_keys = sorted(list(D.keys()))
        self.s_i = s_i
        self.batch = batch

    def privacy_mechanism(self) -> callable:
        """_summary_
            Valid privacy mechanism types: ["None", "GRR", "OUE", "PreHashing"]
        Raises:
            Exception: Invalid privacy mechanism type

        Returns:
            callable: privacy mechanism with given type 
        """

        if self.type == "GRR":
            p = exp(self.varepsilon) / (exp(self.varepsilon)+self.d-1)

            # print(f"Generate Random Response Probability: {p}")
            return self.__GRR(p)
        elif self.type == "GRR_X":
            self.d +=1 
            p = exp(self.varepsilon) / (exp(self.varepsilon)+self.d-1)
            return self.__GRR_X(p)

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
        elif self.type == "OUE":
            return self.__handle_OUE_response()
        else:
            print("Invaild Privacy Type")
    
    def __handle_GRR_response(self):
        def __handle_GRR_response_(responses):
            weight = 1 
            for response in responses:
                if response == None: continue
                
                self.D[response] = self.D.get(response, 0) + weight 
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
                return
            prob = random.random()

            if prob < p:
                return v
            else:
                random_choice_options = list(self.D.keys())
                random_choice_options.remove(v)
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



