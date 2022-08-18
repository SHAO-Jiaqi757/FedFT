from Tree.tree import Node, Tree
class RadixNode(Node):
    def __init__(self, k=None):
        """Initialize Radix Node Struct

        Args:
            k (_type_, optional): key value of the node. Defaults to None.
        """
        super(RadixNode, self).__init__(k)

class RadixTrie(Tree):
    def __init__(self):
        """Initialize the Radix Tree with root
        """
        super(Tree, self).__init__()
        self.root = RadixNode()
        self.__heavyHitters = []

    def insert(self,key, node: RadixNode= None):
        """Expanding the Radix Tree with a new given key value

        Args:
            key (_type_): given key value  
            node (RadixNode, optional): node at which the new node is to be inserted. Defaults to None, the root node.
        """
        if node == None:
            node = self.root 
        self._insert_(key, node)
            
    def _insert_(self,  key, node: RadixNode):
        """Expanding the Radix Tree with a new given key value

        Args:
            node (RadixNode): node at which the new node is to be inserted
            key (String/Integer): given key value
        """
        allSubs = self.getAllSub(key)
        for subKey in allSubs:
            for childKey in node.children.keys():
                if subKey == childKey[:len(subKey)]:
                    if subKey == childKey:
                        if node.children[childKey].isLeaf:
                            node.children[key] = RadixNode(key)
                            node.children[key].isLeaf = True
                            del node.children[childKey]
                            return 
                        self._insert_(key[len(subKey):], node.children[childKey])
                    else: # len(key) < len(childKey)
                        newSubKey = subKey
                        expandKey = childKey[len(newSubKey):]
                        # split node 
                        node.children[newSubKey] = RadixNode(newSubKey)
                        node.children[newSubKey].children[expandKey] = node.children[childKey]
                        node.children[newSubKey].children[expandKey].key = expandKey
                        node.isLeaf = False
                        del node.children[childKey]

                        if subKey == key:
                            node.isLeaf = False
                            node.children[newSubKey].isLeaf = True
                        else:
                            node.children[newSubKey].children[key[len(newSubKey):]] = RadixNode(key[len(newSubKey):])
                            node.children[newSubKey].children[key[len(newSubKey):]].isLeaf = True
                            node.children[newSubKey].isLeaf = False

                    
                    return # find node to be inserted, return
                
        # not find node to be inserted
        node.children[key] = RadixNode(key)
        node.children[key].isLeaf = True
        node.isLeaf = False


    def search(self, target, node: RadixNode= None, display=False):
        """DFS search for target value on Radix Tree

        Args:
            target (_type_): target value that searching for
            node (RadixNode, optional): the radix node that searching procedure starts with. Defaults to None, searching start with root.

        Returns:
            Boolean: True for existing searching path in the tree
        """
        if node == None:
            node = self.root
     
        result = self._search_(node, target)
        display and print(f"Search '{target}': {result}")
        return result

    def _search_(self,node: RadixNode, target):

        """Search a given target value with DFS

        Returns:
            Boolean: True for existing searching path in the tree
        """
        if target == "":
            return node.isLeaf()
        for subKey in self.getAllSub(target):
            if subKey in node.children.keys():
                return self._search_(node.children[subKey], target[len(subKey):])
        return False
    
    def findHeavyHitters(self,node: Node = None):
        """DFS for finding heavy hitters

        Args:
            node (Node, optional): node where DFS starts with. Defaults to None, the root node.
        """
        if node == None:
            node = self.root

        self._display_(node, "", display=False)

    def getHeavyHitters(self):
        self.__heavyHitters = []
        self.findHeavyHitters()
        return self.__heavyHitters

    def display(self, node:RadixNode = None):
        """DFS printing at the start with a given node

        Args:
            node (RadixNode, optional): the radix node that printing procedure starts with. Defaults to None: display starts with root.
        """
        if node == None:
            node = self.root

        self._display_(node, "", display=True)

    def _display_(self, node: RadixNode, s: str ="", display=True):
        """DFS printing at the start with a given node

        Args:
            node (RadixNode): the radix node that printing procedure starts with.
            s (str): current searched result
            display (Boolean): True for printing out DFS results, False for adding heavy hitters
        """
        for childKey in node.children.keys():
            if node.children[childKey].isLeaf:
                if display:
                    print(s + childKey)
                else: 
                    self.__heavyHitters.append(s + childKey)

            else:
                # self._display_(node.children[childKey], s+childKey+"/", display)
                self._display_(node.children[childKey], s+childKey, display)

    def _inference_(self, s):
        pass

    @staticmethod
    def getAllSub(s):
        if type(s) is str:
            for indx in range(len(s), 0, -1):
                yield s[:indx]
