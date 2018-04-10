# coding = utf-8
from collections import defaultdict


class WordHashing(object):
    '''
        Given a corpus and a word in the corpus
        Output the trigram(letter) vector of the word
    '''
    def __init__(self, corpus):
        super(WordHashing, self).__init__()
        self.corpus = corpus
        self.trigram_dict = defaultdict(int)
        self.dict_len = 0
        
        self.construct_dict()
        
    def trigram(self, word):
        ans_list = []
        tmp_word = '#' + word + '#'
        for i in range(len(tmp_word)-2):
            ans_list.append(tmp_word[i:i+3])
        return ans_list

    def construct_dict(self):
        '''
            how to optimize the nesting loops...
        '''
        for word in self.corpus:
            trigram_list = self.trigram(word)
            for trigram in trigram_list:
                self.trigram_dict[trigram] += 1

        # min count
        min_count_keys = []
        for k, v in self.trigram_dict.items():
            if v < 10:
                min_count_keys.append(k)
        for k in min_count_keys:
            del self.trigram_dict[k]

        self.dict_len = len(self.trigram_dict)
        print('self.dict_len:%s' %self.dict_len)
        # convert all dict-values into indexs
        cnt = 0
        for k,v in self.trigram_dict.items():
            self.trigram_dict[k] = cnt
            cnt += 1

    def hashing(self, word):
        '''
            Input: a word
            Output: the hashed (One-Hot)vector of the input word w.r.t the corpus
        '''
        trigram_list = self.trigram(word)
        ans_vec = [0] * self.dict_len
        for trigram in trigram_list:
            try:
                ans_vec[ self.trigram_dict[trigram] ] += 1
            except:
                pass
        return ans_vec