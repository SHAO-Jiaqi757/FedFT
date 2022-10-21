import pickle
import json, heapq, sys
# Trie class
class Node(object):
    def __init__(self, val='', count = 0):
        self.val = val
        self.count = count
        self.par = None
        self.children = {}

class TrieNumeric:
    # init Trie class
    def __init__(self, bits_in_node, k=sys.maxsize):
        self.root = Node()
        self.bits_in_node = bits_in_node
        self.k = k

    def getNode(self):
        return { "children": {}}

    def insert(self, item, count=0, pnode = None, correction=False):
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
            if ch in current.children:                    
                node = current.children[ch]
                node.count += count
            else:
                if correction and current.children:
                    top = sorted(current.children.keys(), key=lambda x: -current.children[x].count)[0]
                    current.children[top].count += 1
                    print("Correction!")
                node = Node(ch, count) 
                current.children[ch] = node  # insert node
                node.par = current

            current = node
            indx += self.bits_in_node

        return current


    def __delete_node(self, node):
        if (not node.par or node.children):
            return
        try:
            del node.par.children[node.val]
            print("DELETE") 
        except KeyError:
            ...
        self.__delete_node(node.par)

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

    def get_predictoin(self, ch, cur, item='', item_list=[]):
        item = item + ch
        if not cur.children and item:
            item_list.append(item)
            return
        ch = sorted(cur.children.keys(), key=lambda x: cur.children[x].count)[-1]
        # for ch in cur.children:
        self.get_predictoin(ch, cur.children[ch], item, item_list)

    def display_trie(self, is_get_hhs = False):
        print("[Trie]:: Items in the trie: ")
        items_list = []
        self.__display_trie('', self.root, '', items_list, is_get_hhs)
        if is_get_hhs:
            return [hhs[0] for hhs in sorted(items_list, key=lambda x: -x[-1])[:self.k]]
        else:
            return items_list

    def __display_trie(self, ch, cur, item='', item_list=[], is_get_hhs = False):
        item = item + ch
        
        if not cur.children and item:
            if is_get_hhs: item_list.append((item, cur.count))
            else: item_list.append(item)
            return
        for ch in cur.children:
            self.__display_trie(ch, cur.children[ch], item, item_list, is_get_hhs)


    def delete(self, item):
        self._delete(self.root, item, 0)

    def _delete(self, current, item, index):
        if(index*self.bits_in_node >= len(item)):
            if current.children:
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

class PriorityQueue(object):
    def __init__(self):
        self.queue = []
 
    def __str__(self):
        return ' '.join([str(i) for i in self.queue])
 
    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
 
    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)
 
    # for popping an element based on Priority
    def delete(self):
        try:
            max_val = 0
            for i in range(len(self.queue)):
                if self.queue[i] > self.queue[max_val]:
                    max_val = i
            item = self.queue[max_val]
            del self.queue[max_val]
            return item
        except IndexError:
            print()
            exit()
 

if __name__ == '__main__':
    trie = TrieNumeric(2)
    pnode = trie.insert('10101')
    trie.insert("10101")
    trie.delete("10101")
    # print()
    print(trie.insert('101', pnode))
    print(trie.insert('111', pnode))
    
    print(trie.display_trie())
