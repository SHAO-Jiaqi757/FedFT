

from abc import ABC, abstractmethod


class EvaluateModuleABC(ABC):
    @abstractmethod
    def NDCG(self, truth_top_k, estimate_top_k):
        pass
    @abstractmethod
    def F1(self, truth_top_k, estimate_top_k):
        pass
    
    def evaluate(self, truth_top_k, estimate_top_k):
        pass