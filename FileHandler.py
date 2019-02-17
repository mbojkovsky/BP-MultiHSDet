from TextProcessor import TextProcessor
import pandas as pd
import numpy as np

class FileHandler():
  def __init__(self):
    self.tp = TextProcessor()

  def extract_multilingual_sentences(self):
    valid_sent_en, train_sent_en = self.extract_sentences('en', write=False, concat=False)
    valid_sent_es, train_sent_es = self.extract_sentences('es', write=False, concat=False)

    valid_sents = np.append(valid_sent_en, valid_sent_es)
    train_sents = np.append(train_sent_en, train_sent_es)
    return valid_sents, train_sents

  def extract_multilingual_labels(self):
    valid_labels_en, train_labels_en = self.extract_labels('en')
    valid_labels_es, train_labels_es = self.extract_labels('es')

    valid_labels = np.append(valid_labels_en, valid_labels_es)
    train_labels = np.append(train_labels_en, train_labels_es)
    return valid_labels, train_labels

  def extract_language_labels(self):
    valid_labels_en, train_labels_en = self.extract_labels('en')
    valid_labels_es, train_labels_es = self.extract_labels('es')

    return [0] * valid_labels_en.shape[0] + [1] * valid_labels_es.shape[0], \
           [0] * train_labels_en.shape[0] + [1] * train_labels_es.shape[0]

  def extract_id(self, lang='en'):
    df = pd.read_csv('./raw_data/test_' + lang + '.tsv', delimiter='\t')
    return df['id'].values

  def extract_test(self, lang='en', write=False):
    df = pd.read_csv('./raw_data/test_' + lang + '.tsv', delimiter='\t')
    df_sent = self.tp.prepare_sentences(df.text, lang)

    if write is True:
      with open('test_data_' + lang, 'w') as file:
        for sent in df_sent:
          file.write('\t'.join(sent) + '\n')

    return df_sent

  def extract_sentences(self, lang='en', write=False, concat=True):
    df = pd.read_csv('./raw_data/dev_' + lang + '.tsv', delimiter='\t')
    tdf = pd.read_csv('./raw_data/train_' + lang + '.tsv', delimiter='\t')

    df = df.drop(columns=['AG', 'TR', 'id', 'HS'])
    tdf = tdf.drop(columns=['AG', 'TR', 'id', 'HS'])

    df_sent = []
    tdf_sent = []

    if concat:
      df = df.append(tdf, ignore_index=False)
      df = df.reset_index(drop=True)
      df_sent = self.tp.prepare_sentences(df.text, lang)

    else:
      df_sent = self.tp.prepare_sentences(df.text, lang)
      tdf_sent = self.tp.prepare_sentences(tdf.text, lang)

    if write and concat:
      with open('text_only_' + lang, 'w') as file:
        for sent in df_sent:
          file.write('\t'.join(sent) + '\n')
    else:
      if concat:
        return df_sent
      return df_sent, tdf_sent

  def extract_labels(self, lang='en'):
    df = pd.read_csv('./raw_data/dev_' + lang + '.tsv', delimiter='\t')
    tdf = pd.read_csv('./raw_data/train_' + lang + '.tsv', delimiter='\t')
    return df['HS'].values, tdf['HS'].values

  def write_file(self, name, content):
    with open(name, 'w') as file:
      for c in content:
        file.write(' '.join([str(x) for x in c]) + '\n')

  def read_file(self, name):
    content = []
    with open(name, 'r') as file:
      for line in file:
        content.append(np.asarray(line.split(' '), dtype=np.float32))
    return np.asarray(content)

