import tensorflow as tf
import numpy as np
from math import ceil
import os

class Model:
  def __init__(self,
               embedder,
               max_len,
               batch_size=100,
               num_epochs=10,
               num_layers=2,
               num_units=256,
               learning_rate=0.001,
               word_dropout=0.4,
               lstm_dropout=0.4,
               adv_lambda=0.5
               ):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.reset_default_graph()

    # hyperparamters
    self.batch_size = batch_size
    self.epochs = np.arange(num_epochs)
    self.learning_rate = learning_rate
    self.num_layers = num_layers
    self.num_units = num_units
    # self.word_dropout = word_dropout
    self.lstm_dropout = lstm_dropout
    self.adv_lambda = adv_lambda

    self.embedder = embedder

    # inputs
    self.x = tf.placeholder(tf.int32, [None, max_len])
    self.y = tf.placeholder(tf.int64, [None])
    self.lang_labels = tf.placeholder(tf.int64, [None])
    self.tokens_length = tf.placeholder(tf.int32, [None])

    # elmo
    self.elmo_init = tf.constant_initializer(self.embedder.weights)
    self.elmo_dic = tf.get_variable(
      name='embedding_weights',
      shape=(self.embedder.weights.shape[0], 1024),
      initializer=self.elmo_init,
      trainable=False)
    self.embedding = tf.nn.embedding_lookup(self.elmo_dic, self.x)

    # multilingual feature extractor for adversarial learning
    self.feature_l1 = tf.layers.dense(self.embedding, 2048, activation=tf.nn.relu)
    self.feature_out = tf.layers.dense(self.feature_l1, 1024, activation=tf.nn.relu)

    # language discriminator
    # dopredu mi ide 1 * self.feature_out; dozadu mi ide -adv_lambda * self.feature_out
    # todo: sem mozno LSTM namiesto dense layeru?
    self.gradient_reversal = tf.stop_gradient((1 + self.adv_lambda) * self.feature_out) - self.adv_lambda * self.feature_out
    self.language_disc_l1 = tf.layers.dense(self.gradient_reversal, 1024, activation=tf.nn.leaky_relu)
    self.language_disc_out = tf.layers.dense(self.language_disc_l1, 2)
    self.disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.language_disc_out,
                                                                                   labels=self.lang_labels))

    # cell and layers definitions
    self.cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_units)
    self.cells_fw = [self.cell() for _ in range(self.num_layers)]
    self.cells_bw = [self.cell() for _ in range(self.num_layers)]

    # LSTM layers with dropout
    self.outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                          self.cells_fw,
                                                                          self.cells_bw,
                                                                          self.feature_out,
                                                                          dtype=tf.float32,
                                                                          sequence_length=self.tokens_length
                                                                          )
    self.dropped_outputs = tf.layers.dropout(self.outputs, self.lstm_dropout)

    # TODO attention here

    """
    # data manipulation before dense layer
    self.pooled = tf.math.reduce_mean(self.dropped_outputs, axis=1)
    self.flat = tf.layers.flatten(self.pooled)

    self.logits = tf.layers.dense(self.flat, units=2)
    """

    self.ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.y))

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.ce_loss)

    self.pred = tf.nn.softmax(self.logits)

    self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), self.y)

    # misc
    self.saver = tf.train.Saver()
    self.init = tf.global_variables_initializer()

 ###
  def train(self, sentences, labels, sentence_lengths, language='en'):
    iterations = np.arange(ceil(len(sentences) / self.batch_size))
    print('Started training!')

    with tf.Session() as sess:
      sess.run(self.init)
      for epoch in self.epochs:
        sen, lab, lng = self.shuffle(sentences, labels, sentence_lengths)
        loss = 0
        for iteration in iterations:
          start = iteration * self.batch_size
          l, _ = sess.run([self.ce_loss, self.optimizer], feed_dict={self.x: sen[start:start+self.batch_size],
                                          self.y: lab[start:start+self.batch_size],
                                          self.tokens_length: lng[start:start+self.batch_size]})
          loss = loss + l
        print('Epoch:', epoch, 'Loss', loss)
      self.saver.save(sess, './chk/model_' + language)
 ###
  def test(self, t_sentences, t_labels, t_sentence_lengths, language='en'):
    iterations = np.arange(ceil(len(t_sentences) / self.batch_size))

    with tf.Session() as sess:
      self.saver.restore(sess, './chk/model_' + language)
      correct = 0
      for iteration in iterations:
        start = iteration * self.batch_size
        corr_pred = sess.run([self.correct_prediction], feed_dict={self.x: t_sentences[start:start+self.batch_size],
                                                                   self.tokens_length: t_sentence_lengths[start:start+self.batch_size],
                                                                   self.y: t_labels[start:start+self.batch_size]})
        correct += corr_pred[0].sum()
    return correct / len(t_sentences)
 
 ###
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
