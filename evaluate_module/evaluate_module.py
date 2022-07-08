from math import log
from typing import Dict, List
from evaluate_module.evaluate_module_abc import EvaluateModuleABC


class EvaluateModule(EvaluateModuleABC):
    def __init__(self,top_k, evaluate_type: str = "NDCG"):
        """_summary_

        Args:
            top_k (int): for `top-k` heavy hitters
            evaluate_type (str, optional): _description_. Defaults to "NDCG".
        """
        self.top_k = top_k
        self.evaluate_type = evaluate_type

    def F1(self, truth_top_k: List, estimate_top_k:List):
        """_summary_

        Args:
            truth_top_k (List): The real top k heavy hitters
            estimate_top_k (List): Estimated top k heavy hitters

        Returns:
            _type_: f1 score
        """
        hit = 0
        not_hit = 0
        for hitter in estimate_top_k:
            if hitter in truth_top_k: hit += 1
            else: not_hit += 1

        f1 = 2*hit/(2*hit + not_hit)
        return f1

    def __DCG(self,estimate_top_k: List, rel:Dict) -> float:
        """_summary_

        Args:
            estimate_top_k (List):Estimated top k heavy hitters
            rel (Dict): relevance of each element in estimate_top_k

        Returns:
            float: Discount Cumulative Gain
            """
        DCG = 0
        for indx, item in enumerate(estimate_top_k):       
            rel_ = 2**rel.get(item, 0) - 1
            DCG += rel_/log(indx+2)
        return DCG
        
    def NDCG(self,truth_top_k, estimate_top_k):
        """_summary_

        Args:
            truth_top_k (list): The real top k heavy hitters
            estimate_top_k (list): Estimated top k heavy hitters
            k (int): top-k heavy hitters
        Returns:
            float: Normalized Discount Cumulative Gain
        """
        top_k = self.top_k
        print(f"Find {len(estimate_top_k)} top-k heavy hitters")
        truth_top_k = truth_top_k[:top_k]
        rel = {}
        for number in truth_top_k:
            rel[number] = top_k 
            top_k-=1
            
        return self.__DCG(estimate_top_k, rel) / self.__DCG(truth_top_k, rel)
            
    def evaluate(self, truth_top_k, estimate_top_k):
        if self.evaluate_type == "NDCG":
            return self.NDCG(truth_top_k, estimate_top_k)
        elif self.evaluate_type == "F1":
            return self.F1(truth_top_k, estimate_top_k)