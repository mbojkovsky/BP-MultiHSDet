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

# TODO niekde robim chybu v odstranovani pismen pri char level zalezitostiach; zatial to netreba riesit
def get_sentences(ft, emb, word_level_processing):
    valid_sent, train_sent = ft.extract_multilingual_sentences()
    valid_labels, train_labels = ft.extract_multilingual_labels()

    if word_level_processing:
        # get token lengths
        valid_token_lengths = [len(sent) for sent in valid_sent]
        train_token_lengths = [len(sent) for sent in train_sent]

        # create indexed sentences
        max_len = max(valid_token_lengths + train_token_lengths)
        valid_sent = emb.create_indexed_sentences(valid_sent, max_len)
        train_sent = emb.create_indexed_sentences(train_sent, max_len)
    else:
        valid_token_lengths = [[len(word) for word in sent] for sent in valid_sent]
        train_token_lengths = [[len(word) for word in sent] for sent in train_sent]

        max_len = 64

        valid_sent = np.array([emb.create_indexed_sentences(sent, max_len) for sent in valid_sent])
        train_sent = np.array([emb.create_indexed_sentences(sent, max_len) for sent in train_sent])

        valid_sent = np.array([x if x.shape != (0, ) else np.array([[0] * max_len]) for x in valid_sent])
        train_sent = np.array([x if x.shape != (0, ) else np.array([[0] * max_len]) for x in train_sent])

        # padding
        pad_len = max([len(x) for x in np.append(valid_sent, train_sent)])
        template = np.zeros(shape=(pad_len, max_len))

        ret = []

        for x in valid_sent:
            tmp = template
            tmp[:x.shape[0], :x.shape[1]] = x
            ret.append(tmp)

        valid_sent = np.array(ret)

        ret = []

        for i, x in enumerate(train_sent):
            tmp = template
            # print(i, x.shape)
            tmp[:x.shape[0], :x.shape[1]] = x
            ret.append(tmp)

        train_sent = np.array(ret)


    return valid_sent, valid_labels, valid_token_lengths, \
           train_sent, train_labels, train_token_lengths

if __name__ == "__main__":

    ft = FileHandler()
    tp = TextProcessor()

    print('Loading embeddings')
    c_emb = e.Embedder(dim=16)
    c_emb.create_embeddings(tp.characters)

    w_emb = e.Embedder(dim=300)
    # w_emb.load_embeddings('muse_dict_en', sep=' ')
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

    print('Loading dataset')

    wv_sent, wv_labs, wv_sent_len, wt_sent, wt_labs, wt_sent_len = get_sentences(ft, w_emb, True)

    # TODO funkciu treba podla mna prekopat
    cv_sent, _, _, ct_sent, _, _ = get_sentences(ft, c_emb, False)

    print('Initializing model')
    batch_size = 64
    num_epochs = 300
    num_layers = 2
    num_units = 256
    learning_rate = 0.003
    word_dropout = 0.4
    lstm_dropout = 0.4

    m = Model(
        w_emb,
        wv_sent.shape[1],
        c_emb,
        cv_sent.shape[2],
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_layers=num_layers,
        num_units=num_units,
        learning_rate=learning_rate,
        lstm_dropout=lstm_dropout)

    # m.train(train_sent, train_labels, train_token_lengths, None,
    #         valid_sent, valid_labels, valid_token_lengths, None)


    m.train(sentences=wt_sent[:10], labels=wt_labs[:10], sentence_lengths=wt_sent_len[:10], words=ct_sent[:10],
            v_sentences=wv_sent[:10], v_labels=wv_labs[:10], v_sentence_lengths=wv_sent_len[:10], v_words=cv_sent[:10])
    # m.train(train_sent[:1000], train_labels[:1000], train_token_lengths[:1000], None,
    #         valid_sent[:1000], valid_labels[:1000], valid_token_lengths[:1000], None)
    print('Done!')
