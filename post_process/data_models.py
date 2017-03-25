from __future__ import division, print_function, unicode_literals
import re
import json
import os


# Define paths to data
module_dir = os.path.dirname(__file__)  # get current directory
bigram_filename = os.path.join(module_dir, 'data/bigram.json')
unigram_filename = os.path.join(module_dir, 'data/unigram.json')
dict_filename = os.path.join(module_dir, 'data/vi_VN.dic')
word_pairs_filename = os.path.join(module_dir, 'data/word_pairs.json')


class BigramModel():
    def __init__(self):
        with open(bigram_filename, 'r', encoding='utf8') as f:
            self.bigram = json.loads(f.read(), encoding='utf8')

    def count(self, word1, word2):
        key = "{} {}".format(word1, word2).lower()
        c = self.bigram.get(key)
        if not c:
            return 0
        else:
            return c


class UnigramModel():
    def __init__(self):
        with open(unigram_filename, 'r', encoding='utf8') as f:
            self.unigram = json.loads(f.read(), encoding='utf8')

    def count(self, word1, word2):
        key = "{} {}".format(word1, word2)
        if not key.istitle():
            key = key.lower()
        c = self.unigram.get(key)
        if not c:
            return 0
        else:
            return c


class Dict():
    def __init__(self):
        # read dictionary file => list of strings
        with open(dict_filename, "r", encoding='utf8') as f:
            self.dict = ' '.join(f.read().split())
    
    def is_in_dict(self, word):
        return re.search(r'\b' + word + r'\b', self.dict, flags=re.IGNORECASE) != None


class WordPairs():
    def __init__(self):
        with open(word_pairs_filename, "r", encoding='utf8') as f:
            self.word_pairs = json.loads(f.read(), encoding='utf8')
        
        self.before = self.word_pairs['before']
        self.after = self.word_pairs['after']
    
    def get_before_list(self, token):
        """
        Return a list of all before tokens of token or None
        """
        key = token.lower()
        return self.before.get(key)

    def get_after_list(self, token):
        """
        Return a list of all after tokens of token or None
        """
        key = token.lower()
        return self.after.get(key)
