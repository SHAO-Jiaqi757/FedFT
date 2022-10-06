import pickle
import json
# Trie class


class Trie:
    # init Trie class
    def __init__(self):
        self.root = self.getNode()

    def getNode(self):
        return {"is_end": False, "children": {}}

    def insert(self, item, pnode = None):
        """_summary_

        Args:
            item (_type_): insert item to trie begin with pnode

        Returns:
            node: end node
        """
        current =  self.root if not pnode else pnode # parent node
        for ch in item:

            if ch in current['children']:
                node = current["children"][ch]
            else:
                node = self.getNode()
                current["children"][ch] = node

            current = node
        current["is_end"] = True
        return current

    def search(self, item):
        current = self.root
        for ch in item:
            if not ch in current['children']:
                return False
            node = current["children"][ch]

            current = node
        return current["is_end"]

    def start_with(self, prefix):
        current = self.root
        for ch in prefix:
            if not ch in current['children']:
                return False
            node = current["children"][ch]

            current = node
        # return True if children contain keys and values
        return bool(current["children"])

    def item_start_with(self, prefix):

        current = self.root
        for ch in prefix:
            if not ch in current['children']:
                print(f"[Trie]::[NO] Items start with [{prefix}]")
                return [] 
            node = current["children"][ch]

            current = node
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
        
        if cur['is_end']: 
            item_list.append(item)

        if not cur["children"]:
            return
        for ch in cur["children"]:
            self.__display_trie(ch, cur['children'][ch], item, item_list)

    def delete(self, item):
        self._delete(self.root, item, 0)

    def _delete(self, current, item, index):
        if(index == len(item)):
            if not current["is_end"]:
                return False
            current["is_end"] = False
            return len(current["children"].keys()) == 0

        ch = item[index]
        if not ch in current['children']:
            return False
        node = current["children"][ch]

        should_delete_current_node = self._delete(node, item, index + 1)

        if should_delete_current_node:
            current["children"].pop(ch)
            return len(current["children"].keys()) == 0

        return False

    def save_to_pickle(self, file_name):
        f = open(file_name + ".pkl", "wb")
        pickle.dump(self.root, f)
        f.close()

    def load_from_pickle(self, file_name):
        f = open(file_name + ".pkl", "rb")
        self.root = pickle.load(f)
        f.close()

    def save_to_json(self, file_name):
        json_data = json.dumps(self.root)
        f = open(file_name + ".json", "w")
        f.write(json_data)
        f.close()

    def load_from_json(self, file_name):
        json_file = open(file_name + ".json", "r")
        self.root = json.load(json_file)
        json_file.close()


if __name__ == '__main__':
    trie = Trie()
    pnode = trie.insert("apple")
    # print()
    print(trie.insert("apsdfk"))
    print(trie.insert("abe"))
    print(trie.item_start_with('ap'))
    # print(trie.display_trie())
