import tensorflow as tf
import numpy as np
from math import ceil
import os

def identity_init(shape, dtype, partition_info=None):
    return tf.eye(shape[0], shape[1], dtype=dtype)

class Model:
  def __init__(self,
               word_embedder,
               max_len,
               batch_size=32,
               num_epochs=10,
               num_layers=2,
               num_units=256,
               attention_size=512,
               learning_rate=0.001,
               lstm_dropout=0.4,
               adv_lambda=0.2,
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

    self.word_embedder = word_embedder

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

    # WORD EMBEDDINGS
    self.dic_init = tf.constant_initializer(self.word_embedder.weights)

    self.word_dic = tf.get_variable(shape=(self.word_embedder.weights.shape[0], self.word_embedder.weights.shape[1]),
                                    initializer=self.dic_init,
                                    trainable=False,
                                    name='Embedding_weight_dict')

    self.embedding = tf.nn.embedding_lookup(self.word_dic,
                                            self.x,
                                            name='Embeddings')

    # CHAR EMBEDDINGS

    # CELL AND LAYER DEFINITIONS

    # CHAR LAYER
    # musim si spravit dasl embedding lookup, nahodne inicializovany
    # prejdem teda cez pismenka v slove
    # nieco mi to vypluje, asi to max poolnem a concatnem vyssie

    # WORD LAYER
    self.cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)

    # multilayer lstm
    self.cells_fw = [self.cell() for _ in range(self.num_layers)]
    self.cells_bw = [self.cell() for _ in range(self.num_layers)]

    self.outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                          self.cells_fw,
                                                                          self.cells_bw,
                                                                          self.embedding,
                                                                          dtype=tf.float32,
                                                                          sequence_length=self.tokens_length
                                                                          )
    # TODO dropout

    # ATTENTION
    # implementacia podla https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    self.word_context = tf.get_variable(name='Att',
                                        shape=[20],
                                        dtype=tf.float32)

    self.att_activations = tf.contrib.layers.fully_connected(self.outputs,
                                                             20,
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
                                                                                 labels=self.y))

    self.c_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.c_train_op = self.c_optimizer.minimize(self.ce_loss)

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

    with tf.Session() as sess:
      # self.saver.restore(sess, './chk/model_' + language)
      sess.run(self.init)
      for epoch in self.epochs:
        self.training = True
        # shuffle inputs
        sen, lab, t_lens = self.shuffle(sentences, labels, sentence_lengths)

        # for loss printing (discriminator and classificator)
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
                                                                   self.y: t_labels[start:end]})
        correct += corr_pred[0].sum()
    return correct / len(t_sentences)
 
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
