from abc import ABC, abstractmethod
class TreeNode(ABC):
    @abstractmethod           
    def isLeaf(self):
        """
        Returns:
            Boolean: Radix Node is leaf or not
        """
        pass
    @abstractmethod 
    def __str__(self):
        """
        Returns:
            String: key value of the radix node
        """
        pass
      

class Tree(ABC):
    inferenceMode: bool = False
    def setInferenceMode(self):
        self.inferenceMode = True
    def resetInferenceMode(self):
        self.inferenceMode = False

    @abstractmethod
    def _insert_(self, key, node):
        pass 
    @abstractmethod
    def insert(self, key, node):
        pass 
    @abstractmethod
    def _search_(self, node, target):
        pass
    @abstractmethod
    def search(self, target, node, display):
        pass 
    @abstractmethod
    def findHeavyHitters(self, node):
        pass 
    @abstractmethod
    def getHeavyHitters(self):
        pass
    @abstractmethod
    def _display_(self, node, s, display):
        pass 
    @abstractmethod
    def display(self, node: TreeNode):
        pass
    @abstractmethod
    def _inference_(self, s):
        pass 
    
    def inference(self, s):
        return self._inference_(s) 





class Node(TreeNode):
    def __init__(self, k):
        self.key = k
        self.children = {}
        # self.isLeaf = 
            
    def isLeaf(self):
        """
        Returns:
            Boolean: Radix Node is leaf or not
        """
        return self.__isLeaf()
        
    def __str__(self):
        """
        Returns:
            String: key value of the radix node
        """
        return self.key
    def __isLeaf(self):
        return not self.children