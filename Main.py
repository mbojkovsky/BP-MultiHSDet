# vykonanie potrebnych operacii nad datasetom
from Model import Model
from FileHandler import FileHandler
import Embedder as e
from hyperopt import fmin, tpe, hp
from TextProcessor import TextProcessor
from sklearn.metrics import f1_score
import numpy as np

valid_token_lengths = []
train_token_lengths = []
max_len = 0

valid_sent = []
train_sent = []

valid_labels = []
train_labels = []

w_emb = None
c_emb = None

def hypertuning_cycle(params):
    m = Model(
        w_emb,
        c_emb,
        max_len,
        batch_size=50,
        num_epochs=6,
        num_layers=params['num_layers'],
        num_units=params['num_units'],
        learning_rate=0.001,
        fcc=True,
        fcc_size=100
    )

    m.train(train_sent, train_labels, train_token_lengths, None, None, None)
    val = m.predict(valid_sent, valid_labels, valid_token_lengths)

    del m
    ret = 1.0 - f1_score(valid_labels, val)
    print(ret)
    return ret

def hypertuning(max_evals):
    space = {
             'num_layers': hp.choice('num_layers', [1, 2]),
             'num_units': hp.choice('num_units', [32, 64, 80, 100, 128]),
             'l1_scale': hp.choice('l1_scale', [0.0, 0.1, 0.2, 0.4]),
             'l2_scale': hp.choice('l2_scale', [0.0, 0.1, 0.2, 0.4])
             }

    best = fmin(hypertuning_cycle, space, algo=tpe.suggest, max_evals=max_evals)
    print(best)

def write(arr, lang='en'):
    with open('result_' + lang, 'w') as file:
        for line in arr:
            file.write(str(line[0]) + '\t' + str(line[1]) + '\n')

def get_sentences(ft, emb):
    valid_sent, train_sent = ft.extract_multilingual_sentences()
    valid_labels, train_labels = ft.extract_multilingual_labels()

    # get token lengths
    valid_token_lengths = [len(sent) for sent in valid_sent]
    train_token_lengths = [len(sent) for sent in train_sent]

    # create indexed sentences
    max_len = max(valid_token_lengths + train_token_lengths)
    valid_sent = emb.create_indexed_sentences(valid_sent, max_len)
    train_sent = emb.create_indexed_sentences(train_sent, max_len)

    return valid_sent, valid_labels, valid_token_lengths, \
           train_sent, train_labels, train_token_lengths

if __name__ == "__main__":

    lang = 'en'
    ft = FileHandler()
    tp = TextProcessor()

    print('Loading embeddings')
    w_emb = e.Embedder(dim=1024)
    w_emb.load_embeddings('dict_' + lang +' +_word', sep=' ')

    c_emb = e.Embedder(dim=1024)
    c_emb.load_embeddings('dict_' + lang + '_char', sep=' ')

    print('Loading dataset')

    valid_sent, train_sent = ft.extract_sentences(lang, concat=False, write=False)
    train_sent = np.append(train_sent, valid_sent, axis=0)

    valid_labels, train_labels = ft.extract_labels(lang)
    train_labels = np.append(train_labels, valid_labels, axis=0)

    valid_sent = ft.extract_test(lang=lang)

    # get token lengths
    valid_token_lengths = [len(sent) for sent in valid_sent]
    train_token_lengths = [len(sent) for sent in train_sent]

    # create indexed sentences
    max_len = max(valid_token_lengths + train_token_lengths)
    valid_sent = w_emb.create_indexed_sentences(valid_sent, max_len)
    train_sent = w_emb.create_indexed_sentences(train_sent, max_len)

    print('################')

    batch_size = 32
    num_epochs = 2
    num_layers = 1
    num_units = 16
    learning_rate = 0.001

    m = Model(
            w_emb,
            c_emb,
            valid_sent.shape[1],
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_layers=num_layers,
            num_units=num_units,
            learning_rate=learning_rate,
            fcc=False,
            fcc_size=32)

    m.train(train_sent, train_labels, train_token_lengths,
                valid_sent, valid_labels, valid_token_lengths)

    print('Done!')
