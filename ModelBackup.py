import tensorflow as tf
import numpy as np
from math import ceil
import os

def identity_init(shape, dtype, partition_info=None):
    return tf.eye(shape[0], shape[1], dtype=dtype)

class Model:
  def __init__(self,
               word_embedder,
               w_max_len,
               char_embedder,
               c_max_len,
               batch_size=32,
               num_epochs=10,
               num_layers=2,
               num_units=256,
               attention_size=512,
               learning_rate=0.001,
               lstm_dropout=0.4,
               adv_lambda=0.2,
               ):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.reset_default_graph()

    # HYPERPARAMETERS
    self.batch_size = batch_size
    self.epochs = np.arange(num_epochs)
    self.learning_rate = learning_rate
    self.num_layers = num_layers
    self.num_units = num_units
    self.lstm_dropout = lstm_dropout
    self.adv_lambda = adv_lambda
    self.attention_size = attention_size

    self.word_embedder = word_embedder
    self.char_embedder = char_embedder
    self.training = False

    # INPUTS
    # Sentence level
    self.sentences = tf.placeholder(tf.int32,
                            [None, w_max_len],
                            name='Sentences')


    self.w_tokens_length = tf.placeholder(tf.int32,
                                        [None],
                                        name='Sentence_lengths')

    # Word level
    self.words = tf.placeholder(tf.int32,
                            [None, w_max_len, c_max_len],
                            name='Words')

    self.labels = tf.placeholder(tf.int64,
                                 [None],
                                 name='HS_labels')
    # WORD EMBEDDINGS
    self.w_dic_init = tf.constant_initializer(self.word_embedder.weights)

    self.word_dic = tf.get_variable(shape=self.word_embedder.weights.shape,
                                    initializer=self.w_dic_init,
                                    trainable=False,
                                    name='Embedding_weight_dict')



    self.w_embedding = tf.nn.embedding_lookup(self.word_dic,
                                            self.sentences,
                                            name='Embeddings')

    # CHAR EMBEDDINGS

    self.pad_emb = tf.get_variable(
        name="pad_embedding",
        initializer=tf.zeros(shape=[1, self.char_embedder.weights.shape[1]], dtype=tf.float32),
        trainable=False)

    self.c_dic_init = tf.constant_initializer(self.char_embedder.weights)

    self.c_dic = tf.get_variable(shape=self.char_embedder.weights.shape,
                                    initializer=self.c_dic_init,
                                    trainable=True,
                                    name='Char_weight_dict')

    print(self.words.shape, 'Input shape')

    self.words_flat = tf.reshape(self.words, shape=[-1, c_max_len])

    print(self.words_flat.shape, 'Reshaped input')

    self.c_embedding = tf.nn.embedding_lookup(tf.concat([self.pad_emb, self.c_dic], axis=0),
                                              self.words_flat,
                                              name='Embeddings')

    print(self.c_embedding.shape, 'Char emb shape')


    # CELL AND LAYER DEFINITIONS
    # CHAR LAYER

    self.char_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                    num_units=128,
                                                    direction='bidirectional',
                                                    dtype=tf.float32,
                                                    name='char_level_lstm')

    self.c_outputs, _ = self.char_lstm(tf.transpose(self.c_embedding,[1, 0, 2]), training=True)
    print(self.c_outputs.shape, 'LSTM OUTPUTS')

    self.c_outputs_mean = tf.math.reduce_mean(tf.transpose(self.c_outputs, [1, 0, 2]), axis=1)
    print(self.c_outputs_mean.shape, 'After reduce_mean') # BATCH x W_MAX_LEN x 128

    self.c_outputs_reshaped = tf.reshape(self.c_outputs_mean, shape=[-1, w_max_len, self.c_outputs.shape[-1]])
    print(self.c_outputs_reshaped.shape, 'Reshaped outputs') # BATCH x W_MAX_LEN x C_max_len x 128

    self.embeddings = tf.concat([self.w_embedding, self.c_outputs_reshaped], axis=2)

    print(self.embeddings.shape, 'CONCAT EMB')


    # WORD LAYER
    self.cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_units)

    self.cells_fw = [self.cell() for _ in range(self.num_layers)]
    self.cells_bw = [self.cell() for _ in range(self.num_layers)]

    self.outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.cells_fw,
                                                                        self.cells_bw,
                                                                        self.embeddings,
                                                                        sequence_length=self.w_tokens_length,
                                                                        dtype=tf.float32,
                                                                        scope='word_level_lstm'
                                                                        )

    print(self.outputs.shape, 'WORD LSTM OUT')

    # ATTENTION
    # implementacia podla https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    self.word_context = tf.get_variable(name='Att',
                                        shape=[w_max_len],
                                        dtype=tf.float32)

    self.att_activations = tf.contrib.layers.fully_connected(self.outputs,
                                                             w_max_len,
                                                             weights_initializer=identity_init)

    self.word_attn_vector = tf.reduce_sum(tf.multiply(self.att_activations,
                                                      self.word_context),
                                          axis=2,
                                          keepdims=True)

    self.att_weights = tf.nn.softmax(self.word_attn_vector,
                                     axis=1)

    self.weighted_input = tf.multiply(self.outputs,
                                      self.att_weights)

    self.pooled = tf.reduce_sum(self.weighted_input,
                                axis=1)

    # TOP-LEVEL SOFTMAX LAYER
    self.logits = tf.layers.dense(self.pooled,
                                  units=2)

    self.ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.labels))

    self.c_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.c_train_op = self.c_optimizer.minimize(self.ce_loss)

    self.pred = tf.nn.softmax(self.logits)

    self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), self.labels)

    # MISC
    self.saver = tf.train.Saver()
    self.init = tf.global_variables_initializer()

  def train(self, sentences, labels, sentence_lengths, words,
            v_sentences, v_labels, v_sentence_lengths, v_words,
            language='en'):
    print('Started training!')

    iterations = np.arange(ceil(sentences.shape[0] / self.batch_size))
    emb = []

    with tf.Session() as sess:
      # self.saver.restore(sess, './chk/model_' + language)
      sess.run(self.init)
      for epoch in self.epochs:
        self.training = True
        # shuffle inputs
        sen, lab, t_lens, char = self.shuffle(sentences, labels, sentence_lengths, words)

        # for loss printing (discriminator and classificator)
        c_loss = 0

        for iteration in iterations:
          start = iteration * self.batch_size
          end = min(start + self.batch_size, sentences.shape[0])

          c_l, _ = sess.run([self.ce_loss, self.c_train_op], feed_dict={self.sentences: sen[start:end],
                                                                        self.labels: lab[start:end],
                                                                        self.words: char[start:end],
                                                                        self.w_tokens_length: t_lens[start:end]
                                                                        })
          c_loss += c_l

        # validation between epochs
        self.saver.save(sess, './chk/model_' + language)
        print('Epoch:', epoch, 'Classifier loss:', c_loss)
        print('TRAIN ACC:', self.test(sentences, labels, sentence_lengths, words, valid=False))
        print('VALID ACC', self.test(v_sentences, v_labels, v_sentence_lengths, v_words), '\n')

  def test(self, t_sentences, t_labels, t_sentence_lengths, t_words, language='en', valid=True):
    iterations = np.arange(ceil(t_sentences.shape[0] / self.batch_size))
    self.training = False

    with tf.Session() as sess:
      self.saver.restore(sess, './chk/model_' + language)
      correct = 0
      loss = 0

      for iteration in iterations:
        start = iteration * self.batch_size
        end = min(start + self.batch_size, t_sentences.shape[0])
        corr_pred, l = sess.run([self.correct_prediction, self.ce_loss], feed_dict={self.sentences: t_sentences[start:end],
                                                                   self.w_tokens_length: t_sentence_lengths[start:end],
                                                                   self.words: t_words[start:end],
                                                                   self.labels: t_labels[start:end]
                                                                   })
        loss += l
        correct += corr_pred.sum()

    if valid is True:
        print(loss)
    return correct / len(t_sentences)
 
  def shuffle(self, sentences, labels, lengths, words):
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)
    shuffled_sentences = []
    shuffled_labels = []
    shuffled_lengths = []
    shuffled_words = []

    for i in indexes:
      shuffled_sentences.append(sentences[i])
      shuffled_labels.append(labels[i])
      shuffled_lengths.append(lengths[i])
      shuffled_words.append(words[i])

    return shuffled_sentences, shuffled_labels, shuffled_lengths, shuffled_words

"""
else:
        max_len = 32

        valid_sent = [[word[:max_len] for word in x] for x in valid_sent]
        train_sent = [[word[:max_len] for word in x] for x in train_sent]

        valid_token_lengths = [[len(word[:max_len]) for word in sent] for sent in valid_sent]
        train_token_lengths = [[len(word[:max_len]) for word in sent] for sent in train_sent]

        valid_sent = np.array([emb.create_indexed_sentences(sent, max_len) for sent in valid_sent])
        train_sent = np.array([emb.create_indexed_sentences(sent, max_len) for sent in train_sent])

        valid_sent = np.array([x if x.shape != (0, ) else np.array([[0] * max_len]) for x in valid_sent])
        train_sent = np.array([x if x.shape != (0, ) else np.array([[0] * max_len]) for x in train_sent])

        # padding
        pad_len = max([len(x) for x in np.append(valid_sent, train_sent)])
        ret = []

        for x in valid_sent:
            tmp = np.zeros(shape=(pad_len, max_len))
            tmp[:x.shape[0], :x.shape[1]] = x
            ret.append(tmp)

        valid_sent = np.array(ret)

        ret = []

        for i, x in enumerate(train_sent):
            tmp = np.zeros(shape=(pad_len, max_len))
            tmp[:x.shape[0], :x.shape[1]] = x
            ret.append(tmp)

        train_sent = np.array(ret)
"""