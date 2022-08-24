# coding=utf-8
# Copyright 2020 The Federated Heavy Hitters AISTATS 2020 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocessing data."""
import collections
import csv
import operator
import pickle
import re
import dict_trie
import numpy as np



def is_valid(word):
    if len(word) < 3 or (word[-1] in [
        '?', '!', '.', ';', ','
    ]) or word.startswith('http') or word.startswith('www'):
        return False

    if word in ['the', 'and', 'for', 'you', 'but']:
        return False
    if re.match(r'^[a-zA-Z]+$', word):
        return True

    return False


def get_clients(filename, dictionary):
    """Returns a dictionary of dictionaries containing per client word frequencies."""

    # read dictionary file
    vocab = dict_trie.Trie()
    words = []
    # clients = {}
    total_clients = 300
    with open(filename, encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if total_clients < 0: break
            total_clients -= 1
            # client = row[4]
            comment = row[5]

            raw_words = comment.lower().split()
            

            raw_words = [re.sub('\W+','', word) for word in raw_words]
            raw_words = [x for x in raw_words if is_valid(x)]
            words = words + [x for x in raw_words if not vocab.has_prefix(x)]


    return words


def truncate_or_extend(word, max_word_len):
    if len(word) > max_word_len:
        word = word[:max_word_len]
    else:
        word += '$' * (max_word_len - len(word))
    return word


def add_end_symbol(word):
    return word + '$'


def generate_triehh_clients(clients):
    clients_num = len(clients)
    triehh_clients = [(clients[i]) for i in range(clients_num)]
    word_freq = collections.defaultdict(lambda: 0)
    for word in triehh_clients:
        word_freq[word] += 1
    word_freq = dict(word_freq)
    with open('clients_triehh.txt', 'wb') as fp:
        pickle.dump(triehh_clients, fp)


def generate_sfp_clients(clients, max_word_len):
    """Generate sfp clients.

    Args:
      clients: input clients.
      max_word_len: maximum word length.
    """
    clients_num = len(clients)
    sfp_clients = [
        truncate_or_extend(clients[i], max_word_len) for i in range(clients_num)
    ]
    word_freq = collections.defaultdict(lambda: 0)
    for word in sfp_clients:
        word_freq[word] += 1
    word_freq = dict(word_freq)
    with open('clients_sfp.txt', 'wb') as fp:
        pickle.dump(sfp_clients, fp)


def main():
    # load dataset from csv file
    # please download dataset from
    # http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
    filename = './dataset/training.1600000.processed.noemoticon.csv'

    # load dictionary
    # please provide your own dictionary if you would like to
    # run out-of-vocabulary experiments
    dictionary = 'dictionary.txt'

    # maximum word length

    clients = get_clients(filename, dictionary)

    with open(f"./dataset/triehh_clients_remove_top5_{len(clients)}.txt", 'wb') as fp:
        pickle.dump(clients, fp)
   

if __name__ == '__main__':

    main()
