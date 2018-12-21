# vykonanie potrebnych operacii nad datasetom
from Model import Model
from FileHandler import FileHandler
import Embedder as e
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

valid_token_lengths = []
train_token_lengths = []
max_len = 0

valid_sent = []
train_sent = []

valid_labels = []
train_labels = []

w_emb = None

def hypertuning(params):
    m = Model(
        w_emb,
        max_len,
        batch_size=params['batch_size'],
        num_epochs=3,
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


if __name__ == "__main__":
    w_emb = e.Embedder(1024, 'elmo_en')
    w_emb.load_embeddings()

    # load dataset
    ft = FileHandler()
    valid_sent, train_sent = ft.extract_sentences('en', write=False, concat=False)
    valid_labels, train_labels = ft.extract_labels('en')
    del ft

    # get token lengths
    valid_token_lengths = [len(sent) for sent in valid_sent]
    train_token_lengths = [len(sent) for sent in train_sent]

    max_len = max(train_token_lengths + valid_token_lengths)
    # create indexed sentences
    valid_sent = w_emb.create_indexed_sentences(valid_sent, max_len)
    train_sent = w_emb.create_indexed_sentences(train_sent, max_len)

    # inicializacia modelu
    # batch_size = 100
    # num_epochs = 10
    # num_layers = 2
    # num_units = 256
    # learning_rate = 0.003
    # word_dropout = 0.4
    # lstm_dropout = 0.4

    space = {'batch_size': hp.choice('batch_size', [50, 100, 150]),
             'num_layers': hp.choice('num_layers', [1, 2]),
             'num_units': hp.choice('num_units', [128, 256, 512]),
             'units3': hp.choice('units3', [64, 512]),
             'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001, 0.003]),
             'word_dropout': hp.choice('word_dropout', [0, 0.2, 0.4, 0.5]),
             'lstm_dropout': hp.choice('lstm_dropout', [0, 0.2, 0.4, 0.5]),
             }

    best = fmin(hypertuning, space, algo=tpe.suggest, max_evals=15)
    print(best)
    # m = Model(
    #     w_emb,
    #     max_len,
    #     batch_size=batch_size,
    #     num_epochs=num_epochs,
    #     num_layers=num_layers,
    #     num_units=num_units,
    #     learning_rate=learning_rate,
    #     word_dropout=word_dropout,
    #     lstm_dropout=lstm_dropout)
    #
    # m.train(train_sent, train_labels, train_token_lengths)
