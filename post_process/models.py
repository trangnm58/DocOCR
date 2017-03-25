from __future__ import division, print_function, unicode_literals
import json
import os
from copy import deepcopy
import difflib

from post_process.data_models import Dict
from post_process.data_models import BigramModel
from post_process.data_models import UnigramModel
from post_process.data_models import WordPairs


# Define paths to data
module_dir = os.path.dirname(__file__)  # get current directory
diacritical_letters = os.path.join(module_dir, 'data/diacritical_letters.json')


class TextProcessor():
    HAS_NONE = 0
    HAS_BEFORE = 1
    HAS_AFTER = 2
    HAS_BEFORE_AFTER = 3
    """
    Separate the raw text into paragraphs and tokenize each paragraphs
    """
    def __init__(self):
        self.before_marks = ['(', '[', '{', '<', '“', '«', '‘', '‹']
        self.after_marks = [')', ']', '}', '>', '”', '»', '’', '›', '?', '.', ',', ';', '!', ':']
        
    def process(self, text):
        paragraphs = self._split_paragraphs(text)  # list of strings
        tokenized_paragraphs = self._tokenize(paragraphs) # list of list of list of tokens
        
        return tokenized_paragraphs
        
    def lists_to_string(self, tokenized_paragraphs):
        text = '\n'.join([' '.join([''.join([t for t in s]) for s in p]) for p in tokenized_paragraphs])
        return text
    
    def get_word_state(self, word):
        """
        word: a list of token [before]middle[after]
        output: state of word and index of middle token
        """
        if len(word) == 1:
            return TextProcessor.HAS_NONE, 0
        elif len(word) == 3:
            return TextProcessor.HAS_BEFORE_AFTER, 1
        elif word[0][0].isalpha():  # first letter of the first token is a letter
            return TextProcessor.HAS_AFTER, 0
        else:
            return TextProcessor.HAS_BEFORE, 1

    def _split_paragraphs(self, text):
        """
        * Delete extra '\n's
        * Find the real '\n's that separate paragraphs, replace all the others with ' '
        * Create a list of paragraphs
        """
        fake_paragraphs = text.split('\n')
        fake_paragraphs = [p.strip() for p in fake_paragraphs if p.strip() != '' ]
        
        if not fake_paragraphs:
            return None
        
        avg_length = sum([len(p) for p in fake_paragraphs]) / len(fake_paragraphs)

        res = []
        res.append(fake_paragraphs[0])
        for i in range(1, len(fake_paragraphs)):
            if len(fake_paragraphs[i-1]) >= avg_length:  # fake paragraph => merge it
                res[len(res) - 1] += " " + fake_paragraphs[i]
            else:
                res.append(fake_paragraphs[i])  # real paragraph
        return res
    
    def _has_letter(self, token):
        for x in token:
            if x.isalpha():
                return True
        return False

    def _all_before_marks(self, token):
        if not token:
            return False
        for i in token:
            if i not in self.before_marks:
                return False
        return True
    
    def _all_after_marks(self, token):
        if not token:
            return False
        for i in token:
            if i not in self.after_marks:
                return False
        return True

    def _tokenize(self, paragraphs):
        if not paragraphs:
            return None
        
        token_para = []
        # tokenize by white spaces
        for p in paragraphs:
            token_para.append(p.split())
        
        # tokenize not letter characters
        for p in token_para:
            for j in range(len(p)):
                if (p[j][0].isalpha() and p[j][-1].isalpha()) or not self._has_letter(p[j]):
                    # a single word token or token does not contain any letter
                    p[j] = [p[j]]
                else:  # [<some>]<letters>[<some>]
                    # find before
                    before = None
                    k = 0
                    while k < len(p[j]) and not p[j][k].isalpha():
                        k += 1
                    if k > 0:
                        before = p[j][:k]
                    # find after
                    after = None
                    l = len(p[j]) - 1
                    while l > 0 and not p[j][l].isalpha():
                        l -= 1
                    if l < len(p[j]) - 1:
                        after = p[j][l+1:]
                    # find middle
                    middle = p[j][k:l+1]
                    
                    keep_before = (before != None and len(before) == 1) or self._all_before_marks(before)
                    keep_after = (after != None and len(after) == 1) or self._all_after_marks(after)              
                    parts = []      
                    if keep_before:
                        parts.append(before)
                    parts.append(middle)
                    if keep_after:
                        parts.append(after)
                    p[j] = parts
                    
        return token_para


# Define global objects
DICT = Dict()
BIGRAM = BigramModel()
UNIGRAM = UnigramModel()
TEXT_PROCESSOR = TextProcessor()
WORD_PAIRS = WordPairs()


class TextBigram():
    """
    A bigram model of text (text is deep_tokenized_paragraphs)
    """
    def __init__(self, tokenized_paragraphs):
        self.bigram = {}
        for p in tokenized_paragraphs:
            for i in range(len(p)-1):
                key = "{} {}".format(p[i], p[i+1]).lower()
                if self.bigram.get(key):
                    self.bigram[key] += 1
                else:
                    self.bigram[key] = 1

    def count(self, word1, word2):
        key = "{} {}".format(word1, word2).lower()
        c = self.bigram.get(key)
        if not c:
            return 0
        else:
            return c


class WordDiacriticCorrector():
    def __init__(self, text):
        # read all vowels file => list of strings
        with open(diacritical_letters, "r", encoding='utf8') as d:
            obj = json.loads(d.read(), encoding='utf8')
        self.diacritical_letters = obj['letter_set']
        
        self.tokenized_paragraphs = TEXT_PROCESSOR.process(text)
    
    def correct(self):
        """
        output: corrected text
        """
        if not self.tokenized_paragraphs:  # nothing to process
            return None

        forward_fix = self._fix_tokenized_paragraphs(direction=1)
        backward_fix = self._fix_tokenized_paragraphs(direction=-1)
        forward_bigram = TextBigram(forward_fix)
        backward_bigram = TextBigram(backward_fix)

        fixed_tokenized_paragraphs = self._choose_best_fix(forward_fix, backward_fix, forward_bigram, backward_bigram)

        return TEXT_PROCESSOR.lists_to_string(fixed_tokenized_paragraphs)

    def _get_word_list(self, word):
        word_set = set()  # no duplicate
        for i in range(len(word)):  # each letter in word
            for l_set in self.diacritical_letters:  # each letter set in all diacritical letters
                if word[i] in l_set:  # letter belongs to letter set
                    new_word = [a for a in word]
                    for l in l_set:  # replace letter with all letter in the set
                        new_word[i] = l
                        word_set.add(''.join(new_word))
                    break  # no need to continue
        # check all words in word_set to see if they exist in dict
        temp = set()
        for w in word_set:
            if DICT.is_in_dict(w):
                temp.add(w)
        temp.add(word)  # make sure the original word is included
        return list(temp)  # convert to list

    def _calculate_P(self, word, near_word, word_list, model, before=True):
        """
        model: self.bigram or self.unigram
        """
        count_list = [0 for _ in word_list]
        for i in range(len(word_list)):
            if before:
                word1 = near_word
                word2 = word_list[i]
            else:
                word1 = word_list[i]
                word2 = near_word
            count_list[i] = model.count(word1, word2)

        if sum(count_list) != 0:
            P_list = [count_list[j] / sum(count_list) for j in range(len(count_list))]
        else:
            P_list = [0 for _ in count_list]
        return P_list

    def _get_best_word(self, word, word_before, word_after):
        """
        input: current word and 2 words next to it
        output: replace, best_word
            replace: True or False
            best_word: is None if replace == False
        """
        word_list = self._get_word_list(word)
        if len(word_list) == 1:  # no replacement at all
            return False, None

        word_idx = word_list.index(word)

        P_bi_before = self._calculate_P(word, word_before, word_list, BIGRAM, before=True)
        P_bi_after = self._calculate_P(word, word_after, word_list, BIGRAM, before=False)

        P_uni_before = self._calculate_P(word, word_before, word_list, UNIGRAM, before=True)
        P_uni_after = self._calculate_P(word, word_after, word_list, UNIGRAM, before=False)

        no_better = sum(P_bi_before) + sum(P_bi_after) == 0
        no_uni = sum(P_uni_before) + sum(P_uni_after) == 0
        if (no_better  # word MAYBE bad but there is no better replacement
            or no_uni):  # no unigram (because the original word is ONLY replaced when there is at least 1 unigram)  
            return False, None

        all_score = [0.25*(P_bi_before[i] + P_bi_after[i]) + 0.25*(P_uni_before[i] + P_uni_after[i]) for i in range(len(word_list))]
        
        # find the index of the biggest score
        best_score = all_score[word_idx]
        best_word = word
        for i in range(len(word_list)):
            score = all_score[i]
            if score > best_score:
                best_score = score
                best_word = word_list[i]
        return True, best_word

    def _fix_tokenized_paragraphs(self, direction=1):
        fix_tokenized_paragraphs = []
        num_of_para = len(self.tokenized_paragraphs)
        
        for j in range(num_of_para):
            temp_p = deepcopy(self.tokenized_paragraphs[j])  # copy
            tracking_fix_token_id = set()
            
            num_of_word = len(temp_p)
            for i in range(num_of_word)[::direction]:
                if i in tracking_fix_token_id:  # i is marked as tracked
                    continue

                # 2 checks to limit the effect of fixed tokens to words around it
                b = i+1 in tracking_fix_token_id
                c = i-1 in tracking_fix_token_id
                
                word_state, idx = TEXT_PROCESSOR.get_word_state(temp_p[i])  # temp_p[i] is a list of tokens
                
                if word_state == TextProcessor.HAS_NONE:
                    if i != 0 and i != len(temp_p) - 1 and not b and not c:
                        replace, best_word = self._get_best_word(temp_p[i][idx], temp_p[i-1][-1], temp_p[i+1][0])
                    elif i != len(temp_p) - 1 and not b:
                        replace, best_word = self._get_best_word(temp_p[i][idx], '', temp_p[i+1][0])
                    elif i != 0 and not c:
                        replace, best_word = self._get_best_word(temp_p[i][idx], temp_p[i-1][-1], '')
                elif word_state == TextProcessor.HAS_BEFORE:
                    if i != len(temp_p) - 1 and not b:
                        replace, best_word = self._get_best_word(temp_p[i][idx], temp_p[i][0], temp_p[i+1][0])
                    else:
                        replace, best_word = self._get_best_word(temp_p[i][idx], temp_p[i][0], '')                        
                elif word_state == TextProcessor.HAS_AFTER:
                    if i != 0 and not c:
                        replace, best_word = self._get_best_word(temp_p[i][idx], temp_p[i-1][-1], temp_p[i][-1])
                    else:
                        replace, best_word = self._get_best_word(temp_p[i][idx], '', temp_p[i][-1])
                else:
                    continue

                if replace:
                    temp_p[i][idx] = best_word  # token in paragraph 'p' index 'i' is set with best word
                    tracking_fix_token_id.add(i)
                    if i != 0 and UNIGRAM.count(temp_p[i-1][-1], temp_p[i][idx]):
                        tracking_fix_token_id.add(i - 1)
                    if i != len(temp_p) - 1 and UNIGRAM.count(temp_p[i][idx], temp_p[i+1][0]):
                        tracking_fix_token_id.add(i + 1)

            fix_tokenized_paragraphs.append(temp_p)
        return fix_tokenized_paragraphs
    
    def _choose_best_fix(self, forward_fix, backward_fix, forward_bigram, backward_bigram):
        """
        run through forward_fix, check each token with backward_fix and replace if it is better
        """
        num_para = len(forward_fix)
        for i in range(num_para):
            num_word = len(forward_fix[i])
            for j in range(num_word):
                if forward_fix[i][j] == backward_fix[i][j]:
                    continue
                
                word_state, idx = TEXT_PROCESSOR.get_word_state(forward_fix[i][j])
                
                if j == 0 and (word_state == TextProcessor.HAS_NONE or word_state == TextProcessor.HAS_BEFORE):  # the first
                    forward_count_bi = forward_bigram.count(forward_fix[i][j][idx], forward_fix[i][j+1][0])
                    backward_count_bi = backward_bigram.count(backward_fix[i][j][idx], backward_fix[i][j+1][0])

                elif j == num_word - 1 and (word_state == TextProcessor.HAS_NONE or word_state == TextProcessor.HAS_AFTER):  # the last
                    forward_count_bi = forward_bigram.count(forward_fix[i][j-1][-1], forward_fix[i][j][idx])
                    backward_count_bi = backward_bigram.count(backward_fix[i][j-1][-1], backward_fix[i][j][idx])

                elif word_state == TextProcessor.HAS_NONE:  # middle
                    forward_count_bi = forward_bigram.count(forward_fix[i][j-1][-1], forward_fix[i][j][idx])  \
                                        + forward_bigram.count(forward_fix[i][j][idx], forward_fix[i][j+1][0])
                    backward_count_bi = backward_bigram.count(backward_fix[i][j-1][-1], backward_fix[i][j][idx])  \
                                        + backward_bigram.count(backward_fix[i][j][idx], backward_fix[i][j+1][0])
                else:
                    continue

                if (backward_count_bi > forward_count_bi
                    or (backward_count_bi == forward_count_bi
                        and backward_fix[i][j] == self.tokenized_paragraphs[i][j])):
                    forward_fix[i][j] = backward_fix[i][j]
                # otherwise, keep forward_fix
        return forward_fix


class WordCorrector():    
    def __init__(self, text):
        self.tokenized_paragraphs = TEXT_PROCESSOR.process(text)
    
    def correct(self):
        for p in self.tokenized_paragraphs:
            num_of_word = len(p)
            for i in range(num_of_word):

                word_state, idx = TEXT_PROCESSOR.get_word_state(p[i])
                if word_state == TextProcessor.HAS_BEFORE_AFTER:
                    continue
                
                if len(p[i][idx]) <= 1:  # word is too short to carry on
                    continue
                
                if not DICT.is_in_dict(p[i][idx]):   # middle token of p[i] needs replacing
                    # find max allow distance
                    word_len = len(p[i][idx])
                    max_dist = int(word_len / 2)
                    
                    letters_only = self._filter_letters_only(p[i][idx])
                    word_list = self._get_word_list(p, i, word_state)
                    
                    best_word = self._get_nearest_word(letters_only, word_list, max_dist)
                    
                    if best_word:
                        p[i][idx] = best_word

        return TEXT_PROCESSOR.lists_to_string(self.tokenized_paragraphs)
                    
    def _get_nearest_word(self, letters_only, word_list, max_dist):
        """
        'word_list': [ [list_word], [list_count] ]
        Return the word in word list that is the most similar to 'letters_only'
        or in other words, have smallest distance from 'letters_only'
        """
        if not word_list[0]:  # there is nothing to replace with
            return None
        
        # find best word
        best_word = word_list[0][0]
        best_dist = self._cal_dist(letters_only, best_word)
        best_count = word_list[1][0]  # consider this when dist == best_dist
        for i in range(len(word_list[0])):
            dist = self._cal_dist(letters_only, word_list[0][i])
            count = word_list[1][i]
            if dist < best_dist or (dist == best_dist and count > best_count):
                best_word = word_list[0][i]
                best_dist = dist
                best_count = word_list[1][i]

        if best_dist > max_dist:  # words are too different from letters_only
            return None
        
        return best_word
    
    def _get_word_list(self, p, i, word_state):
        word_list = [[], []]
        
        merge_list_word = []
        merge_list_count = []
        
        if i < len(p) - 1 and word_state != TextProcessor.HAS_AFTER:  # not the last token
            before_list = WORD_PAIRS.get_before_list(p[i+1][0])
            if before_list:
                before_list_word = before_list[0]
                before_list_count = before_list[1]
                
                merge_list_word += before_list_word
                merge_list_count += before_list_count
                
        if i > 0 and word_state != TextProcessor.HAS_BEFORE:  # not the first token
            after_list = WORD_PAIRS.get_after_list(p[i-1][-1])
            if after_list:
                after_list_word = after_list[0]
                after_list_count = after_list[1]
                
                merge_list_word += after_list_word
                merge_list_count += after_list_count
                
        for i in range(len(merge_list_word)):
            if merge_list_word[i] not in word_list[0]:
                word_list[0].append(merge_list_word[i])
                word_list[1].append(merge_list_count[i])
            else:
                idx = word_list[0].index(merge_list_word[i])
                word_list[1][idx] += merge_list_count[i]

        return word_list
    
    def _filter_letters_only(self, token):
        new_token = ''
        for i in token:
            if i.isalpha():
                new_token += i
        return new_token
    
    def _cal_dist(self, token1, token2):
        """
        distance = 1 ~ 1 replace | 1 add | 1 remove 
        """
        diff = difflib.ndiff(token1.lower(), token2.lower())
        diff_chars = list(diff)
        
        count_replace = 0
        wait_stack = []
        for c in diff_chars:
            if c[0] == ' ':
                wait_stack.append(' ')
            elif c[0] == '-':
                if wait_stack and wait_stack[-1] == '+':
                    del wait_stack[-1]
                    count_replace += 1
                else:
                    wait_stack.append('-')
            elif c[0] == '+':
                if wait_stack and wait_stack[-1] == '-':
                    del wait_stack[-1]
                    count_replace += 1
                else:
                    wait_stack.append('+')
        dist = len([c for c in wait_stack if c != ' ']) + count_replace

        return dist


class PostProcess():
    def process(self, raw_text):
        word_diacritic_corrector = WordDiacriticCorrector(raw_text)
        word_diacritic_corrected_text = word_diacritic_corrector.correct()

        word_corrector = WordCorrector(word_diacritic_corrected_text)
        word_corrected_text = word_corrector.correct()
        
        return word_corrected_text

if __name__ == "__main__":
    text = "Hello, world"
    p = PostProcess()
    res = p.process(text)
    print(res)
    