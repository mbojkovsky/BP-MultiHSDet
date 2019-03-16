# vykonanie potrebnych operacii nad datasetom
from Model import Model
from FileHandler import FileHandler
import Embedder as e
from TextProcessor import TextProcessor

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

    ft = FileHandler()
    tp = TextProcessor()

    print('Loading embeddings')

    w_emb = e.Embedder(dim=300)
    w_emb.load_embeddings('muse_dict_en', sep=' ')
    w_emb.load_embeddings('muse_dict_es', sep=' ')


    print('Loading dataset')


    wv_sent, wv_labs, wv_sent_len, wt_sent, wt_labs, wt_sent_len = get_sentences(ft, w_emb)

    print('Initializing model')
    batch_size = 32
    num_epochs = 10
    num_layers = 1
    num_units = 32
    learning_rate = 0.003

    for _ in range(10):
        m = Model(
        w_emb,
        wv_sent.shape[1],
        use_CNN=False,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_layers=num_layers,
        num_units=num_units,
        learning_rate=learning_rate)

        m.train(sentences=wt_sent, labels=wt_labs, sentence_lengths=wt_sent_len, v_sentences=wv_sent, v_labels=wv_labs, v_sentence_lengths=wv_sent_len)
    print('Done!')
