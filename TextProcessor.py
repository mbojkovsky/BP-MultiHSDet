# nltk
import nltk
from nltk.tokenize import word_tokenize
import re
from functools import reduce

class TextProcessor:
  # nema moc vyznam nic si drzat priamo v pamati
    def __init__(self):
        pass

    def max_word_len(self, sents):
        max_len = 0
        for sent in sents:
            for word in sent:
                tmp = len(word)
                if (tmp > max_len):
                    max_len = tmp
        return max_len
  
    def process_line(self, line):
        # remove url
        tweet = re.sub(r"\b[^ ]*https?://[^ ]*", '', line)

        # remove email, @name
        tweet = re.sub(r"[^ ]*@[^ ]*", '', tweet)
  
        basic_t = nltk.word_tokenize(tweet)

        # vymazem vsetky samostatne symboly
        filtered_lowercased = [w.lower() for w in basic_t if re.match(r'[a-zA-Z0-9]+', w)]

        # replace numbers with <num>
        finished = map(lambda x: re.sub(r"^[.]?([0-9]+[.,]?)+\b", '<number>', x), filtered_lowercased)
        return list(finished)
  
    # prechod datami a aplikacia process_line
    def prepare_sentences(self, data):
        sentences = []

        for i in range(data.size):
            sentences.append(self.process_line(data[i]))
        return sentences
  
    def max_sublist_len(self, ls):
        return len(reduce(lambda a, b: a if len(a) > len(b) else b, ls))
  
    def pad_seq(self, seq, length):
        pad_len = length - len(seq)
        seq = seq + [0] * pad_len
        return seq
