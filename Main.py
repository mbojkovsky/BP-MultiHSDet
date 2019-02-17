from Model import Model
from FileHandler import FileHandler
import Embedder as e

if __name__ == "__main__":
    print('Loading embeddings')
    w_emb = e.Embedder(dim=1024)
    w_emb.load_embeddings('dict_en', sep=' ')
    w_emb.load_embeddings('dict_es', sep=' ')

    # load dataset
    print('Loading dataset')
    ft = FileHandler()
    valid_sent, train_sent = ft.extract_multilingual_sentences()
    valid_labels, train_labels = ft.extract_multilingual_labels()
    valid_lang_labels, train_lang_labels = ft.extract_language_labels()

    # get token lengths
    valid_token_lengths = [len(sent) for sent in valid_sent]
    train_token_lengths = [len(sent) for sent in train_sent]
    max_len = max(valid_token_lengths + train_token_lengths)

    # create indexed sentences
    valid_sent = w_emb.create_indexed_sentences(valid_sent, max_len)
    train_sent = w_emb.create_indexed_sentences(train_sent, max_len)

    # inicializacia modelu
    batch_size = 64
    num_epochs = 100
    num_layers = 2
    num_units = 128
    learning_rate = 0.01
    word_dropout = 0.4
    lstm_dropout = 0.4

    print('Initializing model')
    m = Model(
        w_emb,
        max_len,
        use_CNN=False,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_layers=num_layers,
        num_units=num_units,
        learning_rate=learning_rate,
    )

    # m.train(train_sent[::16], train_labels[::16], train_token_lengths[::16], train_lang_labels[::16],
    #         valid_sent[::16], valid_labels[::16], valid_token_lengths[::16], valid_lang_labels[::16])
    m.train(train_sent[:1000], train_labels[:1000], train_token_lengths[:1000], train_lang_labels[:1000],
            valid_sent[:1000], valid_labels[:1000], valid_token_lengths[:1000], valid_lang_labels[:1000])
    print('Done!')
