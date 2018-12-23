from TextProcessor import TextProcessor
import pandas as pd
import numpy as np

class FileHandler():
  def __init__(self):
    self.tp = TextProcessor()

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

