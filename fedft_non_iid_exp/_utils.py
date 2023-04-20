import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
 
from exp_generate_words import load_words, load_words_count
from Cipher import *

DATA_PATH = "dataset/words_generate/"
def load_clients(clients: list, top_k_words: list, k = sys.maxsize, restriction=-1, encode=True):

    if not encode:
        return top_k_words, clients
        
    for i in range(len(clients)):
        number = encode_word(clients[i])
        clients[i] = number 
    for i in range(len(top_k_words)):
        word = top_k_words[i]
        number = encode_word(word)
        top_k_words[i] = number 
     
    return top_k_words, clients


def encode_file_initate(k=5, filename=""):
    if not filename:
        for n in range(2000, 10001, 1500):
            filename = f"words_generate_{n}"
            
            encode_words(filename, k)
    else:
        encode_words(filename, k)

def encode_words(filename, k: int):
    
    client_path = f"{DATA_PATH}{filename}.txt"
    freq_path = f"{DATA_PATH}{filename}_count.txt"
    
    words = load_words(client_path)
    word_counts = load_words_count(freq_path, k)
    top_k_words = list(word_counts.keys())
    top_k_encode_words, encode_words = load_clients(words, top_k_words=top_k_words, encode=True)
    
    with open(f"{DATA_PATH}{filename}_encode.txt", "w") as f:
        for word in encode_words:
            f.write(str(word) + " ")
            
    with open(f"{DATA_PATH}{filename}_encode_top_{k}.txt", "w") as f:
        for word in top_k_encode_words:
            f.write(str(word) + " ")



def get_non_iid_clusters_topk(top_ks: list, k: int):
    """ Return top k heavy hitters among non-iid clusters.
    Args:
        top_ks (list[list]): top_k hh of cluster i, for index i = 0, 1, 2, ..., cluster i with increasing size
        k (int): desired number of hh

    Returns:
        truth_topk_hh (list): top k hh among all clusters
    """
    truth_topk_hh = []
    while len(truth_topk_hh) < k: 
        for i in range(len(top_ks)-1, -1, -1):
            if len(truth_topk_hh) >= k:
                break
            
            truth_topk_hh.append(top_ks[i].pop(0))
    return truth_topk_hh