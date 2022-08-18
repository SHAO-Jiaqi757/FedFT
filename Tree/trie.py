from Tree.tree import Tree, Node

class TrieNode(Node):
    def __init__(self, weight=0, k=None):
        super(TrieNode, self).__init__(k)
        self.weight = weight


class Trie(Tree):
    def __init__(self):
        """Initialize Trie with root
        """
        super(Tree, self).__init__()
        self.root = TrieNode()
        self.__heavyHitters = []
        self.__inferenceResult = set()
    
    def insert(self, key, weight=0, node:TrieNode = None):
        """Insert a word into Trie

        Args:
            key (): word to be inserted into Trie
            node (TrieNode, optional): node to be inserted. Defaults to None, the root node.
        """
        if node == None:
            node = self.root
        self._insert_(key,weight, node)

 
    def _insert_(self, key, weight, node:TrieNode):
        
        indx = 0 
        while indx < len(key):
            newKey = ord(key[indx]) - ord('a')
            if newKey not in node.children.keys():
                node.children[newKey] = TrieNode(newKey)
                # print(f"Insert:: {key}")
            node = node.children[newKey]
            indx += 1
        

    def search(self, target, node:TrieNode = None, display=False):
        """DFS Search a word in the Trie

        Args:
            target (str): target word
            node (TrieNode, optional): trie node that searching procedure starts with . Defaults to None, the root node.

        Returns:
            Boolean: True for existing searching path in the Trie
        """
        if node is None:
            node = self.root
        result = self._search_(node, target, 0)
        
        display and print(f"Searching {target}: {result}")

        return result

    def _search_(self, node:TrieNode, target, indx:int):
        if indx >= len(target):
            return True
        key = ord(target[indx]) - ord('a')
        if key not in node.children.keys():
            return False 
        return self._search_(node.children[key], target, indx+1)
        
    def findHeavyHitters(self,node:TrieNode = None):
        """DFS for finding heavy hitters

        Args:
            node (TrieNode, optional): node where DFS starts with. Defaults to None, the root node.
        """
        if node == None:
            node = self.root

        self._display_(node, "", display=False)

    def getHeavyHitters(self):
        self.__heavyHitters = []

        self.findHeavyHitters()
        return self.__heavyHitters

    def display(self, node:TrieNode = None):
        """DFS printing at the start with a given node

        Args:
            node (TrieNode, optional): the node that printing procedure starts with. Defaults to None: display starts with root.
        """
        if node == None:
            node = self.root

        self._display_(node, "", display=True)
        
    def _display_(self, node: TrieNode, s, display=True):
        """DFS printing at the start with a given node

        Args:
            node (TrieNode): the node that printing procedure starts with.
            s (): current searched result
        """
        for childKey in node.children.keys():
            if node.children[childKey].isLeaf():
                result = s + chr(childKey + ord('a'))
                if display:
                    print(result)
                else: 
                    if self.inferenceMode:
                        self.__inferenceResult.add(result)
                    else: 
                        self.__heavyHitters.append(result)

            else:
                self._display_(node.children[childKey], s+ chr(childKey + ord('a')), display)
    
    def _inference_(self, s):
        node = self.root
        for indx, i in enumerate(s): 
            if ord(i)-ord('a') not in node.children.keys():
                indx -= 1
                break
            node = node.children[ord(i)-ord('a')]
        if indx >= 0:
            self._display_(node, s[:indx+1], display=False)
        return self.__inferenceResult
    
    def getInferenceResult(self):
        return self.__inferenceResult

    def clearInferenceResult(self):
        self.__inferenceResult = set()
  
