import tensorflow as tf
import numpy as np
from math import ceil
import os

def identity_init(shape, dtype, partition_info=None):
    return tf.eye(shape[0], shape[1], dtype=dtype)

class Model:

  def attention(self, inputs, scope_name):
      word_context = tf.get_variable(name='Att_' + scope_name,
                                     shape=[64],
                                     dtype=tf.float32)

      att_activations = tf.contrib.layers.fully_connected(inputs,
                                                          64,
                                                          weights_initializer=identity_init)

      word_attn_vector = tf.reduce_sum(tf.multiply(att_activations,
                                                        word_context),
                                            axis=2,
                                            keepdims=True)

      att_weights = tf.nn.softmax(word_attn_vector,
                                       axis=1)

      weighted_input = tf.multiply(inputs,
                                        att_weights)

      pooled = tf.reduce_sum(weighted_input,
                            axis=1)

      return pooled

  def __init__(self,
               w_embedder,
               c_embedder,
               max_len,
               fcc,
               fcc_size,
               batch_size=32,
               num_epochs=10,
               num_layers=2,
               num_units=256,
               learning_rate=0.001,
               ):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.reset_default_graph()

    # HYPERPARAMETERS
    self.batch_size = batch_size
    self.epochs = np.arange(num_epochs)
    self.learning_rate = learning_rate
    self.num_layers = num_layers
    self.num_units = num_units

    self.w_embedder = w_embedder
    self.c_embedder = c_embedder

    # INPUTS

    self.training_flag = tf.placeholder(tf.bool)

    self.x = tf.placeholder(tf.int32,
                            [None, max_len],
                            name='Sentences')

    self.y = tf.placeholder(tf.int64,
                            [None],
                            name='HS_labels')

    self.lang_labels = tf.placeholder(tf.int64,
                                      [None],
                                      name='Lang_labels')

    self.tokens_length = tf.placeholder(tf.int32,
                                        [None],
                                        name='Lengths')

    # ELMO
    self.w_elmo_init = tf.constant_initializer(self.w_embedder.weights)

    self.w_elmo_dic = tf.get_variable(shape=self.w_embedder.weights.shape,
                                    initializer=self.w_elmo_init,
                                    trainable=False,
                                    name='Word_Embedding_weight_dict')

    self.w_embedding = tf.nn.embedding_lookup(self.w_elmo_dic,
                                            self.x,
                                            name='Word_level_Embeddings')

    # Char level
    self.c_elmo_init = tf.constant_initializer(self.c_embedder.weights)

    self.c_elmo_dic = tf.get_variable(shape=self.c_embedder.weights.shape,
                                      initializer=self.c_elmo_init,
                                      trainable=False,
                                      name='Embedding_weight_dict')

    self.c_embedding = tf.nn.embedding_lookup(self.c_elmo_dic,
                                              self.x,
                                              name='Embeddings')


    # CELL AND LAYER DEFINITIONS
    self.cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_units)

    self.c_cells_fw = [self.cell() for _ in range(self.num_layers)]
    self.c_cells_bw = [self.cell() for _ in range(self.num_layers)]

    self.c_outputs, _ , _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        self.c_cells_fw,
        self.c_cells_bw,
        self.c_embedding,
        dtype=tf.float32,
        sequence_length=self.tokens_length,
        scope='char_lstm'
    )

    self.c_outputs = self.attention(self.c_outputs, 'char')

    # multilayer lstm
    self.cells_fw = [self.cell() for _ in range(self.num_layers)]
    self.cells_bw = [self.cell() for _ in range(self.num_layers)]

    self.outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                          self.cells_fw,
                                                                          self.cells_bw,
                                                                          self.w_embedding,
                                                                          dtype=tf.float32,
                                                                          sequence_length=self.tokens_length,
                                                                          scope='word_lstm'
                                                                          )

    # ATTENTION
    self.pooled = self.attention(self.outputs, 'word')

    print(self.pooled.shape, 'After attention')

    self.pooled = tf.concat([self.c_outputs, self.pooled],
                             axis=1)

    if fcc is True:
        self.pooled = tf.contrib.layers.fully_connected(self.pooled,
                                                        fcc_size,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                                        weights_initializer=identity_init)

    print(self.pooled.shape, 'After fcc')

    # TOP-LEVEL SOFTMAX LAYER
    self.logits = tf.contrib.layers.fully_connected(self.pooled,
                                                    2,
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.4))

    self.ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.y))
    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    self.c_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.c_train_op = self.c_optimizer.minimize(self.ce_loss)

    self.pred = tf.nn.softmax(self.logits)

    self.ret_pred = tf.argmax(self.pred, 1)

    self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), self.y)

    # MISC
    self.saver = tf.train.Saver()
    self.init = tf.global_variables_initializer()

  def train(self, sentences, labels, sentence_lengths,
            v_sentences, v_labels, v_sentence_lengths,
            language='en'):
    print('Started training!')

    iterations = np.arange(ceil(sentences.shape[0] / self.batch_size))

    with tf.Session() as sess:
      sess.run(self.init)
      for epoch in self.epochs:
        self.training = True
        # shuffle inputs
        sen, lab, t_lens = self.shuffle(sentences, labels, sentence_lengths)

        c_loss = 0

        for iteration in iterations:
          start = iteration * self.batch_size
          end = min(start + self.batch_size, sentences.shape[0])

          c_l, _ = sess.run([self.ce_loss, self.c_train_op], feed_dict={self.x: sen[start:end],
                                                                        self.y: lab[start:end],
                                                                        self.tokens_length: t_lens[start:end],
                                                                        self.training_flag: True})


          c_loss += c_l

        # validation between epochs
        self.saver.save(sess, './chk/model_' + language)
        print('Epoch:', epoch, 'Classifier loss:', c_loss)
        print('TRAIN ACC:', self.test(sentences, labels, sentence_lengths, valid=False))
        print('VALID ACC', self.test(v_sentences, v_labels, v_sentence_lengths, valid=True), '\n')


  def test(self, t_sentences, t_labels, t_sentence_lengths, language='en', valid=True):
    iterations = np.arange(ceil(t_sentences.shape[0] / self.batch_size))
    self.training = False

    with tf.Session() as sess:
      self.saver.restore(sess, './chk/model_' + language)
      correct = 0
      t_loss = 0

      for iteration in iterations:
        start = iteration * self.batch_size
        end = min(start + self.batch_size, t_sentences.shape[0])
        corr_pred, loss = sess.run([self.correct_prediction, self.ce_loss], feed_dict={self.x: t_sentences[start:end],
                                                                                       self.tokens_length: t_sentence_lengths[start:end],
                                                                                       self.y: t_labels[start:end],
                                                                                       self.training_flag: True})
        t_loss += loss
        correct += corr_pred.sum()

    if valid is True:
        print('VALID LOSS', t_loss)
    return correct / len(t_sentences)

  def predict(self, sentences, sentence_lengths, language='en'):
      iterations = np.arange(ceil(sentences.shape[0] / self.batch_size))
      self.training = False
      correct = None

      with tf.Session() as sess:
          self.saver.restore(sess, './chk/model_' + language)

          for iteration in iterations:
              start = iteration * self.batch_size
              end = min(start + self.batch_size, sentences.shape[0])
              pred = sess.run(self.ret_pred,
                              feed_dict={self.x: sentences[start:end],
                                         self.tokens_length: sentence_lengths[start:end],
                                         self.training_flag: False})
              if correct is None:
                correct = np.array(pred)
              else:
                correct = np.append(correct, pred, axis=0)

      return correct

  def shuffle(self, sentences, labels, lengths):
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)
    shuffled_sentences = []
    shuffled_labels = []
    shuffled_lengths = []

    for i in indexes:
      shuffled_sentences.append(sentences[i])
      shuffled_labels.append(labels[i])
      shuffled_lengths.append(lengths[i])

    return shuffled_sentences, shuffled_labels, shuffled_lengths
