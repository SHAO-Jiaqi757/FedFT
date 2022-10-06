import pickle
import json
# Trie class


class TrieNumeric:
    # init Trie class
    def __init__(self, bits_in_node):
        self.root = self.getNode()
        self.bits_in_node = bits_in_node

    def getNode(self):
        return { "children": {}}

    def insert(self, item, pnode = None):
        """_summary_

        Args:
            item (_type_): insert item to trie begin with pnode

        Returns:
            node: end node
        """
        
        if pnode is None:
            item_len = len(item)
            if item_len % self.bits_in_node != 0:
                item = ('0' * ((item_len // 2 + 1)*self.bits_in_node - item_len)) + item
        else: 
            print("Inserting item in wrong bit-length!")
            return pnode
        current =  self.root if not pnode else pnode # parent node

        indx = 0
        while indx < len(item):
            ch = item[indx: indx +self.bits_in_node]
            if ch in current['children']:
                node = current["children"][ch]
            else:
                node = self.getNode()
                current["children"][ch] = node

            current = node
            indx += self.bits_in_node
        return current

    def search(self, item):
        current = self.root
        indx = 0 
        while indx < len(item):
            ch = item[indx: indx +self.bits_in_node]
            if not ch in current['children']:
                return False
            node = current["children"][ch]

            current = node
            indx += self.bits_in_node
        return self.is_end(current)

    def start_with(self, prefix):
        current = self.root
        indx = 0 
        while indx < len(prefix):
            ch = prefix[indx: indx +self.bits_in_node]
            if not ch in current['children']:
                return False
            node = current["children"][ch]

            current = node
            indx += self.bits_in_node
        # return True if children contain keys and values
        return bool(current["children"])
        
    def is_end(self, node):
        return not node["children"]

    def item_start_with(self, prefix):

        current = self.root
        indx = 0
        while indx < len(prefix):
            ch = prefix[indx: indx +self.bits_in_node]
            if not ch in current['children']:
                print(f"[Trie]::[NO] Items start with [{prefix}]")
                return [] 
            node = current["children"][ch]

            current = node
            indx += self.bits_in_node
        # return True if children contain keys and values
        items_list = []
        print(f'[Trie]::Items start with [{prefix}]')
        self.__display_trie('', current, prefix, items_list)
        return items_list
  
    def display_trie(self):
        print("[Trie]:: Items in the trie: ")
        items_list = []
        self.__display_trie('', self.root, '', items_list)
        return items_list

    def __display_trie(self, ch, cur, item='', item_list=[]):
        item = item + ch
        
        if not cur["children"]:
            item_list.append(item)
            return
        for ch in cur["children"]:
            self.__display_trie(ch, cur['children'][ch], item, item_list)


    def delete(self, item):
        self._delete(self.root, item, 0)

    def _delete(self, current, item, index):
        if(index*self.bits_in_node == len(item)):
            if current["children"]:
                return False
            return self.is_end(current) 

        ch = item[index*self.bits_in_node: (index+1)*self.bits_in_node]
        if not ch in current['children']:
            return False
        node = current["children"][ch]

        should_delete_current_node = self._delete(node, item, index + 1)

        if should_delete_current_node:
            current["children"].pop(ch)
            return len(current["children"].keys()) == 0

        return False

if __name__ == '__main__':
    trie = TrieNumeric(2)
    pnode = trie.insert('10101')
    # print()
    print(trie.insert('101', pnode))
    print(trie.insert('111', pnode))
    
    print(trie.display_trie())
