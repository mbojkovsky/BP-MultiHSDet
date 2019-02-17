# vykonanie potrebnych operacii nad datasetom
from Model import Model
from FileHandler import FileHandler
import Embedder as e
from hyperopt import fmin, tpe, hp
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
    print('Loading embeddings')
    w_emb = e.Embedder(dim=1024)
    w_emb.load_embeddings('dict_en', sep=' ')
    w_emb.load_embeddings('dict_es', sep=' ')

    """
    print(w_emb.weights[1])
    print(w_emb.word2idx['the'])
    with open('dict_en', 'w') as file:
        for i, x in zip(w_emb.weights[1:], w_emb.word2idx):
            file.write(x + ' ')
            file.write(' '.join([str(k) for k in i]))
            file.write('\n')
    exit(0)
    """

    # load dataset
    print('Loading dataset')
    ft = FileHandler()
    valid_sent, train_sent = ft.extract_multilingual_sentences()
    valid_labels, train_labels = ft.extract_multilingual_labels()
    valid_lang_labels, train_lang_labels = ft.extract_language_labels()
    # print(len(valid_labels), len(train_lang_labels), valid_sent.shape, train_sent.shape, valid_labels.shape, train_labels.shape)

    # get token lengths
    valid_token_lengths = [len(sent) for sent in valid_sent]
    train_token_lengths = [len(sent) for sent in train_sent]
    max_len = max(valid_token_lengths + train_token_lengths)
    #
    # create indexed sentences
    valid_sent = w_emb.create_indexed_sentences(valid_sent, max_len)
    train_sent = w_emb.create_indexed_sentences(train_sent, max_len)

    # inicializacia modelu
    batch_size = 64
    num_epochs = 300
    num_layers = 2
    num_units = 256
    learning_rate = 0.002
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

    # m_train_sent = np.append(train_sent[:100], train_sent[-100:], axis=0)
    # m_train_labels = np.append(train_labels[:100], train_labels[-100:])
    # m_train_token_lengths = np.append(train_token_lengths[:100], train_token_lengths[-100:])
    # m_train_lang_labels = np.append(train_lang_labels[:100], train_lang_labels[-100:])

    # print(sum(m_train_lang_labels) / len(m_train_lang_labels), m_train_lang_labels.shape)

    # m.train(m_train_sent, m_train_labels, m_train_token_lengths, m_train_lang_labels,
    #         valid_sent[::100], valid_labels[::100], valid_token_lengths[::100], valid_lang_labels[::100])

    # m.train(train_sent[:1000], train_labels[:1000], train_token_lengths[:1000], train_lang_labels[:1000],
    #         valid_sent[:1000], valid_labels[:1000], valid_token_lengths[:1000], valid_lang_labels[:1000])

    m.train(train_sent, train_labels, train_token_lengths, train_lang_labels,
            valid_sent, valid_labels, valid_token_lengths, valid_lang_labels)
    print('Done!')
