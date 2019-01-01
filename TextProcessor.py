from nltk.tokenize import word_tokenize
import re
from functools import reduce
from spacy.lang.es import Spanish
from spacy.lang.en import English

class TextProcessor:
  # nema moc vyznam nic si drzat priamo v pamati
    def __init__(self):
        self.nlp_en = English()
        self.nlp_es = Spanish()

    def max_word_len(self, sents):
        max_len = 0
        for sent in sents:
            for word in sent:
                tmp = len(word)
                if (tmp > max_len):
                    max_len = tmp
        return max_len
  
    def process_line(self, line, lang='en'):
        # remove url
        tweet = re.sub(r"\b[^ ]*https?://[^ ]*", '', line)

        # remove email, @name
        tweet = re.sub(r"[^ ]*@[^ ]*", '', tweet)

        basic_t = []
        if lang == 'en':
            basic_t = [str(token) for token in self.nlp_en(tweet) if not token.is_stop]
        else:
            basic_t = [str(token) for token in self.nlp_es(tweet) if not token.is_stop]

        # vymazem vsetky samostatne symboly
        filtered_lowercased = [w.lower() for w in basic_t if re.match(r'[a-zA-Z0-9]+', w)]

        # replace numbers with <number>
        finished = list(map(lambda x: re.sub(r"^[.]?([0-9]+[.,]?)+\b", '<number>', x), filtered_lowercased))
        return finished
  
    # prechod datami a aplikacia process_line
    def prepare_sentences(self, data, lang='en'):
        sentences = []

        for i in range(data.size):
            sentences.append(self.process_line(data[i], lang=lang))
        return sentences
  
    def max_sublist_len(self, ls):
        return len(reduce(lambda a, b: a if len(a) > len(b) else b, ls))
  
    def pad_seq(self, seq, length):
        pad_len = length - len(seq)
        seq = seq + [0] * pad_len
        return seq
