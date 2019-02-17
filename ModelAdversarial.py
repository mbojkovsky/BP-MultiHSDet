import tensorflow as tf
import numpy as np
from math import ceil
import os

def identity_init(shape, dtype, partition_info=None):
    return tf.eye(shape[0], shape[1], dtype=dtype)


def adversarial_l(f_out, adv_lambda):
    # LANGUAGE DISCRIMINATOR
    # dopredu mi ide 1 * self.feature_out; dozadu mi ide -adv_lambda * self.feature_out

    gradient_reversal = tf.stop_gradient((1 + adv_lambda) * f_out) - adv_lambda * f_out

    # vytvorim 1024 rozmernu reprezentaciu pre kazdy jeden tweet
    # self.d_pooled = tf.math.reduce_mean(self.gradient_reversal,
    #                                     axis=1)

    # fully connected layer
    language_disc_l1 = tf.contrib.layers.fully_connected(
        gradient_reversal,
        1024,
        weights_initializer=identity_init,
        weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.1))

    # znizim dimenzionalitu na 2 (kvoli dvom pouzitym jazykom)
    language_disc_out = tf.layers.dense(language_disc_l1,
                                             units=2)

    return language_disc_out


class Model:
  def attention(self, inputs):
      return NotImplementedError

  def __init__(self,
               embedder,
               max_len,
               batch_size=32,
               num_epochs=10,
               num_layers=2,
               num_units=256,
               attention_size=512,
               learning_rate=0.001,
               lstm_dropout=0.4,
               adv_lambda=0.25,
               disc_units=1024,
               training=True
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
    self.training = training

    self.embedder = embedder

    # INPUTS
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
    self.elmo_init = tf.constant_initializer(self.embedder.weights)

    self.elmo_dic = tf.get_variable(shape=(self.embedder.weights.shape[0], 1024),
                                    initializer=self.elmo_init,
                                    trainable=False,
                                    name='Embedding_weight_dict')

    embedding = tf.nn.embedding_lookup(self.elmo_dic,
                                            self.x,
                                            name='Embeddings')

    # MULTILINGUAL FEATURE EXTRACTOR


    self.f_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_units)
    self.f_cells_fw = [self.f_cell() for _ in range(self.num_layers)]
    self.f_cells_bw = [self.f_cell() for _ in range(self.num_layers)]

    self.f_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            self.f_cells_fw,
            self.f_cells_bw,
            embedding,
            dtype=tf.float32,
            sequence_length=self.tokens_length
            )

    self.word_context = tf.get_variable(name='Att',
                                   shape=[25],
                                   dtype=tf.float32)

    self.att_activations = tf.contrib.layers.fully_connected(self.f_outputs,
                                                        25,
                                                        weights_initializer=identity_init)

    self.word_attn_vector = tf.reduce_sum(tf.multiply(self.att_activations,
                                                      self.word_context),
                                     axis=2,
                                     keepdims=True)

    self.att_weights = tf.nn.softmax(self.word_attn_vector,
                                axis=1)

    self.weighted_input = tf.multiply(self.f_outputs,
                                 self.att_weights)

    self.f_out = tf.reduce_sum(self.weighted_input,
                           axis=1)


    self.d1 = adversarial_l(self.f_out, self.adv_lambda)

    # loss v diskriminatore
    self.disc1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.d1,
                                                                                   labels=self.lang_labels))

    # CELL AND LAYER DEFINITIONS
    # self.cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
    #
    # multilayer lstm
    # self.cells_fw = [self.cell() for _ in range(self.num_layers)]
    # self.cells_bw = [self.cell() for _ in range(self.num_layers)]
    #
    # self.outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
    #                                                                       self.cells_fw,
    #                                                                       self.cells_bw,
    #                                                                       self.f_out,
    #                                                                       dtype=tf.float32,
    #                                                                       sequence_length=self.tokens_length
    #                                                                       )

    # self.pooled = tf.math.reduce_mean(self.outputs,
    #                                   axis=1)

    self.l1 = tf.contrib.layers.fully_connected(self.f_out,
                                                       512,
                                                       weights_initializer=identity_init,
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    self.pooled = tf.contrib.layers.fully_connected(self.l1,
                                                256,
                                                weights_initializer=identity_init,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    self.d2 = adversarial_l(self.pooled, 0.2)

    # loss v diskriminatore
    self.disc2_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.d2,
                                                                                    labels=self.lang_labels))

    # TOP-LEVEL SOFTMAX LAYER
    self.logits = tf.layers.dense(self.pooled,
                                  units=2)

    self.ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.y))

    self.c_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.c_train_op = self.c_optimizer.minimize(self.ce_loss)

    self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    self.d1_train_op = self.d_optimizer.minimize(self.disc1_loss)

    self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    self.d2_train_op = self.d_optimizer.minimize(self.disc2_loss)

    self.pred = tf.nn.softmax(self.logits)

    self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), self.y)

    # MISC
    self.saver = tf.train.Saver()
    self.init = tf.global_variables_initializer()

  def train(self, sentences, labels, sentence_lengths, lang_labels,
            v_sentences, v_labels, v_sentence_lengths, v_lang_labels,
            language='en'):
    print('Started training!')

    iterations = np.arange(ceil(sentences.shape[0] / self.batch_size))
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session() as sess:
      sess.run(self.init)
      for epoch in self.epochs:
        self.training = True
        # shuffle inputs
        sen, lab, t_lens, l_lab = self.shuffle(sentences, labels, sentence_lengths, lang_labels)

        # for loss printing (discriminator and classificator)
        d1_loss = 0
        d2_loss = 0
        c_loss = 0

        for iteration in iterations:
          start = iteration * self.batch_size
          end = min(start + self.batch_size, sentences.shape[0])

          c_l, _,  = sess.run([self.ce_loss, self.c_train_op], feed_dict={self.x: sen[start:end],
                                                                        self.y: lab[start:end],
                                                                        self.lang_labels: l_lab[start:end],
                                                                        self.tokens_length: t_lens[start:end]})

          d2_l, _ = sess.run([self.disc2_loss, self.d2_train_op], feed_dict={self.x: sen[start:end],
                                                                             self.lang_labels: l_lab[start:end],
                                                                             self.tokens_length: t_lens[start:end]})

          d1_l, _ = sess.run([self.disc1_loss, self.d1_train_op], feed_dict={self.x: sen[start:end],
                                                                          self.lang_labels: l_lab[start:end],
                                                                          self.tokens_length: t_lens[start:end]})

          c_loss += c_l
          d1_loss += d1_l
          d2_loss += d2_l

        # validation between epochs
        self.saver.save(sess, './chk/model_' + language)
        print('Epoch:', epoch, 'Classifier loss:', c_loss, 'Discriminator loss:', d1_loss, 'Top-D loss:', d2_loss)
        print('TRAIN ACC:', self.test(sentences, labels, sentence_lengths, lang_labels))
        print('VALID ACC', self.test(v_sentences, v_labels, v_sentence_lengths, v_lang_labels), '\n')


  def test(self, t_sentences, t_labels, t_sentence_lengths, t_lang_labels, language='en'):
    iterations = np.arange(ceil(t_sentences.shape[0] / self.batch_size))
    self.training = False

    with tf.Session() as sess:
      self.saver.restore(sess, './chk/model_' + language)
      correct = 0

      for iteration in iterations:
        start = iteration * self.batch_size
        end = min(start + self.batch_size, t_sentences.shape[0])
        corr_pred = sess.run([self.correct_prediction], feed_dict={self.x: t_sentences[start:end],
                                                                   self.tokens_length: t_sentence_lengths[start:end],
                                                                   self.y: t_labels[start:end],
                                                                   self.lang_labels: t_lang_labels[start:end]})
        correct += corr_pred[0].sum()
    return correct / t_sentences.shape[0]
 
  def shuffle(self, sentences, labels, lengths, langs):
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)
    shuffled_sentences = []
    shuffled_labels = []
    shuffled_lengths = []
    shuffled_langs = []
  
    for i in indexes:
      shuffled_sentences.append(sentences[i])
      shuffled_labels.append(labels[i])
      shuffled_lengths.append(lengths[i])
      shuffled_langs.append(langs[i])
    
    return shuffled_sentences, shuffled_labels, shuffled_lengths, shuffled_langs
