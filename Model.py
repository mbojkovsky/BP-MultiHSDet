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

    self.c_dic = tf.get_variable('char_embeddings',
                                        shape=[self.char_embedder.weights.shape[0], self.char_embedder.weights.shape[1]],
                                        trainable=True)

    self.c_embedding = tf.nn.embedding_lookup(self.c_dic,
                                              self.words)

    # self.c_embedding = tf.one_hot(self.words,
    #                               depth=self.char_embedder.weights.shape[0])

    print(self.c_embedding.shape, 'C EMB') # BATCH x W_MAX_LEN x C_MAX_LEN x EMB_SIZE

    # CELL AND LAYER DEFINITIONS
    # CHAR LAYER
    self.c_cell = lambda: tf.contrib.rnn.LSTMBlockCell(64)
    self.c_cells_fw = [self.c_cell() for _ in range(1)]
    self.c_cells_bw = [self.c_cell() for _ in range(1)]

    # self.c_flat = tf.reshape(self.c_embedding, shape=[-1, w_max_len, c_max_len * self.char_embedder.weights.shape[1]])

    self.c_flat = tf.reshape(self.c_embedding, shape=[-1, c_max_len, self.char_embedder.weights.shape[1]])
    print(self.c_flat.shape, 'C EMB AFTER RESHAPE') # BATCH * W_MAX_LEN x C_MAX_LEN x EMB_SIZE

    self.c_outputs, self.fw, self.bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        self.c_cells_fw,
        self.c_cells_bw,
        self.c_flat,
        dtype=tf.float32,
        scope='char_level_lstm')

    # self.c_outputs_mean = tf.concat([self.fw[0][0], self.bw[0][0]], axis=-1)
    self.c_outputs_mean = tf.math.reduce_mean(self.c_outputs, axis=1)
    print(self.c_outputs_mean.shape) # BATCH * W_MAX_LEN x 128

    self.c_outputs_reshaped = tf.reshape(self.c_outputs_mean, shape=[-1, w_max_len, self.c_outputs_mean.shape[1]])
    print(self.c_outputs_reshaped.shape) # BATCH x W_MAX_LEN x 128

    # concat char level features with extracted features from char level lstm

    """
    self.w_emb_dropped = tf.layers.dropout(self.w_embedding,
                                           rate=0.5,
                                           training=self.training)

    self.embedding = tf.concat([self.w_emb_dropped, self.c_outputs_reshaped],
                               -1)

    # WORD LAYER
    self.cell = lambda: tf.contrib.rnn.LSTMBlockCell(num_units)
    self.cells_fw = [self.cell() for _ in range(self.num_layers)]
    self.cells_bw = [self.cell() for _ in range(self.num_layers)]

    self.outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.cells_fw,
                                                                        self.cells_bw,
                                                                        self.embedding,
                                                                        dtype=tf.float32,
                                                                        sequence_length=self.w_tokens_length,
                                                                        scope='word_level_lstm'
                                                                        )

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
"""
    self.todo = tf.layers.dense(tf.layers.flatten(self.c_outputs_reshaped), units=256)

    # TOP-LEVEL SOFTMAX LAYER
    self.logits = tf.layers.dense(self.todo,
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
        print('TRAIN ACC:', self.test(sentences, labels, sentence_lengths, words))
        print('VALID ACC', self.test(v_sentences, v_labels, v_sentence_lengths, v_words), '\n')

  def test(self, t_sentences, t_labels, t_sentence_lengths, t_words, language='en'):
    iterations = np.arange(ceil(t_sentences.shape[0] / self.batch_size))
    self.training = False

    with tf.Session() as sess:
      self.saver.restore(sess, './chk/model_' + language)
      correct = 0

      for iteration in iterations:
        start = iteration * self.batch_size
        end = min(start + self.batch_size, t_sentences.shape[0])
        corr_pred = sess.run([self.correct_prediction], feed_dict={self.sentences: t_sentences[start:end],
                                                                   self.w_tokens_length: t_sentence_lengths[start:end],
                                                                   self.words: t_words[start:end],
                                                                   self.labels: t_labels[start:end]
                                                                   })
        correct += corr_pred[0].sum()
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
