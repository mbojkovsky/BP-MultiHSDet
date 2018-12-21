from TextProcessor import TextProcessor
import pandas as pd
import numpy as np

class FileHandler():
  def __init__(self):
    self.tp = TextProcessor()

  def extract_sentences(self, language='en', write=False, concat=True):
    df = pd.read_csv('./raw_data/dev_' + language + '.tsv', delimiter='\t')
    tdf = pd.read_csv('./raw_data/train_' + language + '.tsv', delimiter='\t')

    df = df.drop(columns=['AG', 'TR', 'id', 'HS'])
    tdf = tdf.drop(columns=['AG', 'TR', 'id', 'HS'])

    df_sent = []
    tdf_sent = []

    if concat:
      df = df.append(tdf, ignore_index=False)
      df = df.reset_index(drop=True)
      df_sent = self.tp.prepare_sentences(df.text)

    else:
      df_sent = self.tp.prepare_sentences(df.text)
      tdf_sent = self.tp.prepare_sentences(tdf.text)

    if write and concat:
      with open('text_only_' + language, 'w') as file:
        for sent in df_sent:
          file.write('\t'.join(sent) + '\n')
    else:
      if concat:
        return df_sent
      return df_sent, tdf_sent

  def extract_labels(self, language='en'):
    df = pd.read_csv('./raw_data/dev_' + language + '.tsv', delimiter='\t')
    tdf = pd.read_csv('./raw_data/train_' + language + '.tsv', delimiter='\t')
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

