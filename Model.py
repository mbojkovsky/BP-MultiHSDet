import tensorflow as tf
import numpy as np
from math import ceil
import os

def gs(a):
  print(a.shape)

class Model:
  def __init__(self,
               embedder,
               max_len,
               batch_size=100,
               num_epochs=10,
               num_layers=2,
               num_units=256,
               attention_size=512,
               learning_rate=0.001,
               word_dropout=0.4,
               lstm_dropout=0.4,
               adv_lambda=0.5
               ):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.reset_default_graph()

    # HYPERPARAMETERS
    self.batch_size = batch_size
    self.epochs = np.arange(num_epochs)
    self.learning_rate = learning_rate
    self.num_layers = num_layers
    self.num_units = num_units
    # self.word_dropout = word_dropout
    self.lstm_dropout = lstm_dropout
    self.adv_lambda = adv_lambda
    self.attention_size = attention_size

    self.embedder = embedder

    # INPUTS
    self.x = tf.placeholder(tf.int32, [None, max_len])
    self.y = tf.placeholder(tf.int64, [None])
    self.lang_labels = tf.placeholder(tf.int64, [None])
    self.tokens_length = tf.placeholder(tf.int32, [None])

    # ELMO
    self.elmo_init = tf.constant_initializer(self.embedder.weights)
    self.elmo_dic = tf.get_variable(
      name='embedding_weights',
      shape=(self.embedder.weights.shape[0], 1024),
      initializer=self.elmo_init,
      trainable=False)
    self.embedding = tf.nn.embedding_lookup(self.elmo_dic, self.x)

    # MULTILINGUAL FEATURE EXTRACTOR
    self.cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(256)
    self.f_cells_fw = [self.cell() for _ in range(1)]
    self.f_cells_bw = [self.cell() for _ in range(1)]

    with tf.variable_scope('feature_extractor'):
      self.f_outputs, self.f_output_state_fw, self.f_output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        self.f_cells_fw,
        self.f_cells_bw,
        self.embedding,
        dtype=tf.float32,
        sequence_length=self.tokens_length
      )

    # LANGUAGE DISCRIMINATOR
    # todo: vyskusat treba aj CNN
    # todo: nejaky rozumnejsi pooling mozno najst
    # dopredu mi ide 1 * self.feature_out; dozadu mi ide -adv_lambda * self.feature_out
    self.gradient_reversal = tf.stop_gradient((1 + self.adv_lambda) * self.f_outputs) - self.adv_lambda * self.f_outputs
    # vytvorim 1024 rozmernu reprezentaciu pre kazdy jeden tweet
    self.d_pooled = tf.math.reduce_mean(self.gradient_reversal, axis=1)
    # fully connected layer
    self.language_disc_l1 = tf.layers.dense(self.d_pooled, 1024, activation=tf.nn.leaky_relu)
    # znizim dimenzionalitu na 2 (kvoli dvom pouzitym jazykom)
    self.language_disc_out = tf.layers.dense(self.language_disc_l1, 2)
    # loss v diskriminatore
    self.disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.language_disc_out,
                                                                                   labels=self.lang_labels))

    # CELL AND LAYER DEFINITIONS
    # multilayer lstm
    self.cells_fw = [self.cell() for _ in range(self.num_layers)]
    self.cells_bw = [self.cell() for _ in range(self.num_layers)]

    self.outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                          self.cells_fw,
                                                                          self.cells_bw,
                                                                          self.f_outputs,
                                                                          dtype=tf.float32,
                                                                          sequence_length=self.tokens_length
                                                                          )
    # TODO mozno sem dropout?

    # ATTENTION
    # implementacia podla https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    self.word_context = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1), trainable=True)
    # jeden fully connected layer
    self.att_activations = tf.layers.dense(self.outputs, self.attention_size, activation=tf.nn.tanh)
    # vynasobenie s word_contextom
    self.word_act = tf.tensordot(self.att_activations, self.word_context, axes=1)
    # vypocitame attention koeficienty
    self.coefs = tf.nn.softmax(self.word_act)
    # aplikujeme koeficienty na output
    self.out_with_att = self.outputs * tf.expand_dims(self.coefs, -1)
    # poolneme
    self.pooled = tf.reduce_sum(self.out_with_att, 1)

    # TOP-LEVEL SOFTMAX LAYER
    self.flat = tf.layers.flatten(self.pooled)

    self.logits = tf.layers.dense(self.flat, units=2)
    self.ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                 labels=self.y))

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.c_train_op = self.optimizer.minimize(self.ce_loss)
    self.d_train_op = self.optimizer.minimize(self.disc_loss)
    self.train_ops = tf.group(self.c_train_op, self.d_train_op)

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
      sess.run(self.init)
      for epoch in self.epochs:
        # shuffle inputs
        sen, lab, t_lens, l_lab = self.shuffle(sentences, labels, sentence_lengths, lang_labels)

        # for loss printing (discriminator and classificator)
        d_loss = 0
        c_loss = 0

        for iteration in iterations:
          start = iteration * self.batch_size
          end = min(start + self.batch_size, sentences.shape[0])

          c_l, d_l, _ = sess.run([self.ce_loss, self.disc_loss, self.train_ops], feed_dict={self.x: sen[start:end],
                                                                                            self.y: lab[start:end],
                                                                                            self.tokens_length: t_lens[start:end],
                                                                                            self.lang_labels: l_lab[start:end]})
          c_loss += c_l
          d_loss += d_l

        # validation between epochs
        self.saver.save(sess, './chk/model_' + language)
        print('Epoch:', epoch, 'Classifier loss:', c_loss, 'Discriminator loss:', d_loss)
        print('TRAIN ACC:', self.test(sentences, labels, sentence_lengths, lang_labels))
        print('VALID ACC', self.test(v_sentences, v_labels, v_sentence_lengths, v_lang_labels), '\n')


  def test(self, t_sentences, t_labels, t_sentence_lengths, t_lang_labels, language='en'):
    iterations = np.arange(ceil(t_sentences.shape[0] / self.batch_size))

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
    return correct / len(t_sentences)
 
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
