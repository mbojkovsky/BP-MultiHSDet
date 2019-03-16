import tensorflow as tf
import numpy as np
from math import ceil
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def identity_init(shape, dtype, partition_info=None):
    return tf.eye(shape[0], shape[1], dtype=dtype)

def print_scores(str, pred, real):
    print(str, ' acc: ', accuracy_score(real, pred))
    print(str, ' rec: ', recall_score(real, pred))
    print(str, ' pre: ', precision_score(real, pred))
    print(str, ' f1: ', f1_score(real, pred))

class Model:

  def attention(self, inputs, scope_name):
      word_context = tf.get_variable(name='Att_' + scope_name,
                                     shape=[16],
                                     dtype=tf.float32)

      att_activations = tf.contrib.layers.fully_connected(inputs,
                                                          16,
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
               max_len,
               use_CNN,
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

    # INPUTS
    self.x = tf.placeholder(tf.int32,
                            [None, max_len],
                            name='Sentences')

    self.y = tf.placeholder(tf.int64,
                            [None],
                            name='HS_labels')

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

    # if use_CNN is True:
    #     self.conv = tf.layers.conv1d(self.w_embedding,
    #                                  filters=100,
    #                                  kernel_size=4,
    #                                  activation=tf.nn.relu,
    #                                 )
    #
    #     self.lstm_input = tf.layers.max_pooling1d(self.conv,
    #                                           pool_size=4,
    #                                           strides=4)
    #
    #     mode = 'unidirectional'
    #
    # else:

    self.lstm_input = self.w_embedding
    mode = 'bidirectional'

    self.lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers,
                                                    num_units=num_units,
                                                    direction=mode,
                                                    dtype=tf.float32,
                                                    name='lstm')

    self.outputs, _ = self.lstm(tf.transpose(self.lstm_input,
                                                  [1, 0, 2]))

    # if use_CNN is True:
    self.pooled = tf.reduce_max(tf.transpose(self.outputs,
                                                 [1, 0, 2]),
                                axis=1)

    # else:
    #     self.pooled = self.attention(tf.transpose(self.outputs,
    #                                               [1, 0, 2]),
    #                                  'lstm')



    # TOP-LEVEL SOFTMAX LAYER
    self.logits = tf.contrib.layers.fully_connected(self.pooled,
                                                    2,
                                                    weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=0.1,
                                                                                                            scale_l2=0.1))

    self.ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.y))

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
                                                                        self.tokens_length: t_lens[start:end]})


          c_loss += c_l

        # validation between epochs
        self.saver.save(sess, './chk/model_' + language)
        print('Epoch:', epoch, 'Classifier loss:', c_loss)
        self.test(sentences, labels, sentence_lengths, valid=False)
        self.test(v_sentences, v_labels, v_sentence_lengths, valid=True)

        print()

  def test(self, t_sentences, t_labels, t_sentence_lengths, language='en', valid=True):
    iterations = np.arange(ceil(t_sentences.shape[0] / self.batch_size))
    self.training = False

    with tf.Session() as sess:
      self.saver.restore(sess, './chk/model_' + language)
      correct = np.array([])
      t_loss = 0

      for iteration in iterations:
        start = iteration * self.batch_size
        end = min(start + self.batch_size, t_sentences.shape[0])
        corr_pred, loss = sess.run([self.ret_pred, self.ce_loss],
                                   feed_dict={self.x: t_sentences[start:end],
                                              self.tokens_length: t_sentence_lengths[start:end],
                                              self.y: t_labels[start:end]})
        t_loss += loss
        correct = np.append(correct, corr_pred)

      # model evaluation
      if valid is True:
        print('Classifier loss on valid:', t_loss)
        print_scores('VALID', correct, t_labels)

      else:
        print_scores('TEST', correct, t_labels)
 
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
