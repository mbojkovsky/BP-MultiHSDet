from nltk.tokenize import word_tokenize
import re
import numpy as np
from functools import reduce
from spacy.lang.es import Spanish
from spacy.lang.en import English
from collections import defaultdict
import emoji
import string

class TextProcessor:
    def __init__(self):
        self.nlp_en = English()
        self.nlp_es = Spanish()
        self.characters = "0123456789abcdefghijklmnopqrstuvwxyz!' " + \
                          '#$%&()*+,-./:;<=>?@[\]^_`{|}~áéíóú¿¡üñçå¿¡€¢£¥°âãäåïðñöü‡œ‰”„'
        self.count = []

    def get_char_list(self):
        return self.characters

    def max_word_len(self, sents):
        max_len = 0
        for sent in sents:
            for word in sent:
                tmp = len(word)
                if (tmp > max_len):
                    max_len = tmp
        return max_len
  
    def process_line(self, line, word_processing, lang='en'):
        tweet = re.sub(r"\b[^ ]*https?://[^ ]*", '', line.lower())

        # remove email, @name
        tweet = re.sub(r"[^ ]*@[^ ]*", '', tweet)

        # fix malformed hashtags
        tweet = re.sub('#', ' ', tweet)
        tweet = re.sub(r' {2}', ' ', tweet)

        # fix html characters
        tweet = re.sub(r'&gt;', '>', tweet)
        tweet = re.sub(r'&lt;', '<', tweet)
        tweet = re.sub(r'&le;', '<=', tweet)
        tweet = re.sub(r'&ge;', '>=', tweet)

        # change emoji to word representations
        tweet = emoji.demojize(tweet, delimiters=('', ''))
        tweet = re.sub(r'_', ' ', tweet)

        # tokenization
        basic_t = []
        if lang == 'en':
            # remove non ASCII for English
            if word_processing:
                tweet = re.sub(r"[^\x00-\x7f]", '', tweet)
            else:
                tweet = ''.join(list(filter(lambda x: x in self.characters, tweet)))
            basic_t = [str(token) for token in self.nlp_en(tweet) if not token.is_stop]
        else:
            # for spanish remove every character not from spanish character set
            tweet = ''.join(list(filter(lambda x: x in self.characters, tweet)))
            basic_t = [str(token) for token in self.nlp_es(tweet) if not token.is_stop]

        # vymazem vsetky samostatne symboly
        filtered_lowercased = [w for w in basic_t if re.match(r'[a-zA-Z0-9]+', w)]

        # replace numbers with number
        finished = list(map(lambda x: re.sub(r"^[.]?([0-9]+[.,]?)+\b", 'number', x), filtered_lowercased))

        finished = list(map(lambda x: x[:32], finished))

        self.count.append(len(finished))

        # max 50 slov
        return finished[:50]

    # prechod datami a aplikacia process_line
    def prepare_sentences(self, data, lang='en', word_level=True):
        sentences = []

        for i in range(data.size):
            sentences.append(self.process_line(data[i], word_processing=word_level, lang=lang))
        return sentences

    def extract_character_counts(self, text):
        ret = defaultdict(int)
        for char in ''.join(np.hstack(text)):
            ret[char] += 1
        return ret

    def max_sublist_len(self, ls):
        return len(reduce(lambda a, b: a if len(a) > len(b) else b, ls))
  
    def pad_seq(self, seq, length):
        pad_len = length - len(seq)
        seq = seq + [0] * pad_len
        return seq
