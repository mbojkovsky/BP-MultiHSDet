# vykonanie potrebnych operacii nad datasetom
from Model import Model
from FileHandler import FileHandler
import Embedder as e
from hyperopt import fmin, tpe, hp
from TextProcessor import TextProcessor
import numpy as np

valid_token_lengths = []
train_token_lengths = []
max_len = 0

valid_sent = []
train_sent = []

valid_labels = []
train_labels = []

w_emb = None

def hypertuning_cycle(params):
    m = Model(
        w_emb,
        max_len,
        batch_size=params['batch_size'],
        num_epochs=10,
        num_layers=params['num_layers'],
        num_units=params['num_units'],
        learning_rate=params['learning_rate'],
        word_dropout=params['word_dropout'],
        lstm_dropout=params['lstm_dropout'],
    )

    m.train(train_sent, train_labels, train_token_lengths)
    val = m.test(valid_sent, valid_labels, valid_token_lengths)
    del m
    return val

def hypertuning(max_evals):
    # TODO cele prekopat
    space = {'batch_size': hp.choice('batch_size', [50, 100, 150]),
             'num_layers': hp.choice('num_layers', [1, 2]),
             'num_units': hp.choice('num_units', [128, 256, 512]),
             'units3': hp.choice('units3', [64, 512]),
             'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001, 0.003]),
             'word_dropout': hp.choice('word_dropout', [0, 0.2, 0.4, 0.5]),
             'lstm_dropout': hp.choice('lstm_dropout', [0, 0.2, 0.4, 0.5]),
             }

    best = fmin(hypertuning_cycle, space, algo=tpe.suggest, max_evals=max_evals)
    print(best)


if __name__ == "__main__":

    ft = FileHandler()
    tp = TextProcessor()

    print('Loading embeddings')
    c_emb = e.Embedder(dim=16)
    c_emb.create_embeddings(tp.characters)

    w_emb = e.Embedder(dim=300)
    w_emb.load_embeddings('muse_dict_en', sep=' ')
    w_emb.load_embeddings('muse_dict_es', sep=' ')


    # train_s, valid_s = ft.extract_sentences('es', write=False, concat=False)
    # matrix = []
    #
    # sentences = np.append(train_s, valid_s, axis=0)
    # words = []
    # print(sentences.shape)
    #
    # for sent in sentences:
    #   for word in sent:
    #     idx = w_emb.word2idx.pop(word, None)
    #     if idx is not None:
    #         matrix.append(w_emb.weights[idx])
    #         words.append(word)
    #
    # matrix = np.asarray(matrix)
    # words = np.asarray(words)
    # print(matrix.shape, words.shape)
    #
    # with open('muse_dict_es', 'w') as file:
    #     for emb, word in zip(matrix, words):
    #         file.write(word + ' ')
    #         file.write(' '.join([str(k) for k in emb]))
    #         file.write('\n')

    # load dataset
    print('Loading dataset')
    valid_sent, train_sent = ft.extract_multilingual_sentences()
    valid_labels, train_labels = ft.extract_multilingual_labels()
    # print(len(valid_labels), len(train_lang_labels), valid_sent.shape, train_sent.shape, valid_labels.shape, train_labels.shape)

    # get token lengths
    valid_token_lengths = [len(sent) for sent in valid_sent]
    train_token_lengths = [len(sent) for sent in train_sent]
    max_len = max(valid_token_lengths + train_token_lengths)

    # create indexed sentences
    valid_sent = w_emb.create_indexed_sentences(valid_sent, max_len)
    train_sent = w_emb.create_indexed_sentences(train_sent, max_len)

    print(train_sent[0])
    # inicializacia modelu
    batch_size = 64
    num_epochs = 100
    num_layers = 2
    num_units = 256
    learning_rate = 0.003
    word_dropout = 0.4
    lstm_dropout = 0.4

    print('Initializing model')
    m = Model(
        w_emb,
        max_len,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_layers=num_layers,
        num_units=num_units,
        learning_rate=learning_rate,
        lstm_dropout=lstm_dropout)


    m.train(train_sent, train_labels, train_token_lengths, None,
            valid_sent, valid_labels, valid_token_lengths, None)
    # m.train(train_sent[::100], train_labels[::100], train_token_lengths[::100], None,
    #         valid_sent[::100], valid_labels[::100], valid_token_lengths[::100], None)
    # m.train(train_sent[:1000], train_labels[:1000], train_token_lengths[:1000], None,
    #         valid_sent[:1000], valid_labels[:1000], valid_token_lengths[:1000], None)
    print('Done!')
