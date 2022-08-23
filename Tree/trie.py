from tree import Tree, Node

class TrieNode(Node):
    def __init__(self, k, weight=0):
        """
        TrieNode:: key, children(List[TrieNode]), weight(float, optional), level(int), prefix(ancesters' keys, optinal)

        Args:
            weight (float, optional): node's weight, used for weighted_trie. Defaults to 0.
            k (_type_, optional): key value with fixed bit length. Defaults to None.
        """
        super(TrieNode, self).__init__(k)
        self.weight = weight
        self.level = 0


class Trie(Tree):
    def __init__(self, bit_len: int):

        super(Tree, self).__init__()
        self.bit_len = bit_len
        
        self.root = TrieNode(0)
        self.nodes = [{0: self.root}] # Nodes in the Trie, {key: TrieNode}
        self.level = 0
        self.__heavyHitters = []
        self.__inferenceResult = set()
    
    def insert(self, pattern, batch, weight=0, node:TrieNode = None):
        """Insert a word into TrieNode

        Args:
            parttern (): parttern to be inserted into Trie
            node (TrieNode, optional): node to be inserted. Defaults to None, the root node.
        """

 
        if node == None:
            node = self.root
        while self.level <= batch:
            self.nodes.append({})
            self.level+=1
        self._insert_(pattern, weight, node, batch)

 
    def _insert_(self, pattern, weight, node:TrieNode, batch):
        
        offset = batch
        while offset >= 0:
            newKey = pattern >> (offset*self.bit_len)
            newKey = (newKey & (1<<self.bit_len)-1)
            
            if newKey not in node.children.keys():
                newNode = TrieNode(newKey)
                self.nodes[node.level+1][newKey] = newNode # append newNode at this level.
                newNode.level = node.level+1
                node.children[newKey] = newNode

                node = newNode
            else:
                node = node.children[newKey]
                
            offset-=1
        

    def search(self, target, node:TrieNode = None, display=False):
        """DFS Search a word in the Trie

        Args:
            target (str): target word
            node (TrieNode, optional): trie node that searching procedure starts with . Defaults to None, the root node.

        Returns:
            TrieNode: last node in existing searching path in the Trie
        """
        if node is None:
            node = self.root
        cur, result = self._search_(node, 0, target, self.level-1)
        
        not display and print(f"Searching {target}: {cur}")

        return cur, result

    def _search_(self, node:TrieNode, cur, target, offset:int):
        if offset < 0:
            return cur, node

        key = (target >> (offset*self.bit_len)) & ((1<<self.bit_len) - 1)
        if key not in node.children.keys():
            return cur, node 
        
        node = node.children[key]

        cur = (cur << self.bit_len) + key
        if node.isLeaf() & offset > 0:  # 
            if (cur << (offset*self.bit_len)) == target:
                return cur, node

        return self._search_(node, cur, target, offset-1)
        
    def __findHeavyHitters(self, node:TrieNode = None):
        """DFS for finding heavy hitters

        Args:
            node (TrieNode, optional): node where DFS starts with. Defaults to None, the root node.
        """
        if node == None:
            node = self.root

        self._display_(node, 0, display=False)

    def getHeavyHitters(self):
        self.__heavyHitters = []

        self.__findHeavyHitters()
        return self.__heavyHitters

    def display(self, node:TrieNode = None):
        """DFS printing at the start with a given node

        Args:
            node (TrieNode, optional): the node that printing procedure starts with. Defaults to None: display starts with root.
        """
        if node == None:
            node = self.root

        self._display_(node, 0, display=True)
        
    def _display_(self, node: TrieNode, s, display=True):
        """DFS printing at the start with a given node

        Args:
            node (TrieNode): the node that printing procedure starts with.
            s (): current searched result
        """
        for childKey in node.children.keys():
            if node.children[childKey].isLeaf():
                result = (s << self.bit_len) + childKey
                offset =  self.level - node.children[childKey].level
                result = result << (self.bit_len*offset)
                
                if display:
                    print(result)
                else: 
                    if self.inferenceMode:
                        self.__inferenceResult.add(result)
                    else: 
                        self.__heavyHitters.append(result)

            else:
                self._display_(node.children[childKey], (s << self.bit_len) + childKey, display)
    
    def __inference(self, s):

        cur, node = self.search(s)

        self._display_(node, cur, display=False)
        return self.__inferenceResult
    
    def getInferenceResult(self):
        return self.__inferenceResult

    def clearInferenceResult(self):
        self.__inferenceResult = set()
  
    def inference(self, s):
        self.inferenceMode = True
        result = self.__inference(s)
        self.inferenceMode = False
        return result

