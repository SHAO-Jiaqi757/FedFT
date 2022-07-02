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

    def privacy_mechanism(self):
        if self.type == "GRR":
            if not dict:
                raise Exception("Dict is required for GRR")

            d = len(self.D)
            p = exp(self.varepsilon) / (exp(self.varepsilon)+d-1)

            # print(f"Generate Random Response Probability: {p}")

            return self.__GRR(p)
        elif self.type == "None":
            return lambda x: x
    
    def __handle_GRR_response(self):
        def __handle_GRR_response_(responses):
            for response in responses:
                if self.D.get(response, -1) != -1:
                    self.D[response] += 1
            return self.D

        return __handle_GRR_response_

    def handle_response(self):
        if self.type == "GRR":
            return self.__handle_GRR_response()
        elif self.type == "None":
            return self.__handle_GRR_response()

