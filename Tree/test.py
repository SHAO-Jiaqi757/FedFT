import random
from trie import Trie

if __name__ == '__main__':
    bit_len = 2
    max_bit_len = 8
    iterations = max_bit_len//bit_len
    tree = Trie(bit_len)
    tree.insert(8, 2)
    tree.insert(1, 2)
    tree.insert(10, 2)
    tree.insert(9, 2)

    tree.search(8)
    tree.search(9)
    tree.search(10)
    tree.display()

    # infer  = tree.inference(2)
    # print(infer)
    hh = tree.getHeavyHitters()
    print(hh)


    