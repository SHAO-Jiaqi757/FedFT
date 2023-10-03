from collections import Counter
import random, os
from wonderwords import RandomWord
import numpy as np
import matplotlib.pyplot as plt
from utils import encode_words

PATH = "dataset/words_generate/"
def encode_file_initate(k=5, filename="", datapath=PATH):
    if not filename:
        for n in range(2000, 10001, 1500):
            filename = f"words_generate_{n}"
            
            encode_words(filename, k, datapath)
    else:
        encode_words(filename, k, datapath)

def generate_random_word(count=5000, save_path=PATH+"words_generate.txt"):
    r = RandomWord()
    # generate a list of 2000 random words
    words = r.random_words(amount=count, include_parts_of_speech=["nouns"])
    # filter out the words that are not nouns

    # write the words to a file
    with open(save_path, "w") as f:
        for word in words:
            f.write(word + " ") 
            
            
def load_words(file_path=PATH+"words_generate.txt"):
    with open(file_path, "r") as f:
        words = f.read().strip().split(" ")
    return words

def load_words_count(file_path, top_k=-1):
    # file {word: count} sorted by count
    with open(file_path, "r") as f:
        # top_k lines
        words = f.readlines()
        if top_k != -1:
            words = words[:top_k]
        words = [word.split(":") for word in words]
        words = {word[0]: int(word[1]) for word in words}
    
    return words

def generate_words_with_frequency(words, sample_n_word, total_n_words):
    """randomly sample sample_n_word words. Using Zipf's distribution to generate the frequency of the words.
    Show the frequency distribution of the sampled words
    """
    # sample the words
    sampled_words = random.choices(words, k=sample_n_word)
     

    # generate frequency for the words with Zipf's distribution such that
    word_freq = []
    for i in range(len(sampled_words)):
        # word_freq.append(1/(i+1)**1.5) #  the $j^{th}$ most frequent value has a frequency proportional to $\frac{1}{j^{1.5}}$ 
        word_freq.append(1/(i+1))
    word_freq = np.array(word_freq)
    word_freq = word_freq / np.sum(word_freq)
    
    # sample the words with the frequency
    sampled_words = np.random.choice(sampled_words, size=total_n_words, replace=True, p=word_freq) 

    # show the frequency distribution
    # plt.hist(sampled_words)
    # plt.xticks(rotation=90)     # roate x labels to vertical
    # plt.show()
    
    return sampled_words
            

if __name__ == "__main__":
    
    # check if PATH exists
    if not os.path.exists(PATH): 
        os.makedirs(PATH) 
        print("Directory " , PATH ,  " Created ")
    
    generate_random_word()
     
    words = load_words()

    for total_n_words in range(2000, 10001, 1500):
        sampled_words = generate_words_with_frequency(words, 5000, total_n_words)
        print("sampled {} words".format(total_n_words))
        # Top 5 frequent words with count {words: count}
        word_count = Counter(sampled_words)
         
        print("Top 5 frequent words with count:", word_count.most_common(5))
        print("\n\n")
        with open(f"{PATH}/words_generate_{total_n_words}.txt", "w") as f:
            for word in sampled_words:
                f.write(word + " ") 
                
        # write {word: count} to file sorted by count
        with open(f"{PATH}/words_generate_{total_n_words}_count.txt", "w") as f:
            for word, count in word_count.most_common():
                f.write(f"{word}: {count}  \n") 
   
        encode_file_initate(k) 