from math import exp
import random
import time
from typing import Dict, List

from privacy_module.privacy_module_abs import PrivacyModuleABC

random.seed(time.time_ns())
class PrivacyModule(PrivacyModuleABC):
    def __init__(self, varepsilon: float, D: Dict = {}, type: str = "GRR"):
        self.varepsilon = varepsilon
        self.D = D
        self.type = type
        self.D_keys = sorted(list(D.keys()))


    def privacy_mechanism(self) -> callable:
        """_summary_
            Invalid privacy mechanism types: ["GRR", "None", "OUE"]
        Raises:
            Exception: Invalid privacy mechanism type

        Returns:
            callable: privacy mechanism with given type 
        """

        if self.type == "GRR":
            if not dict:
                raise Exception("Dict is required for GRR")

            d = len(self.D)
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d-1)

            # print(f"Generate Random Response Probability: {p}")

            return self.__GRR(p)
        elif self.type == "None":
            return lambda x: x
        elif self.type == "OUE":
            return self.__OUE()
        

    def handle_response(self) -> callable:
        """_summary_
            Invalid privacy mechanism types: ["GRR", "None", "OUE"]
        Returns:
            callable: response handler with given type
        """
        if self.type == "GRR":
            return self.__handle_GRR_response()
        elif self.type == "None":
            return self.__handle_GRR_response()
        elif self.type == "OUE":
            return self.__handle_OUE_response()
    
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
            prob = random.random()

            if prob < p:
                return v
            else:
                return random.choice([number for number in self.D if number != v])
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
            response_aggregate = [sum(x) for x in zip(responses)]
            for index, count in enumerate(response_aggregate):
                key = self.D_keys[index]
                self.D[key] += count
            return self.D
        return __handle_OUE_response_