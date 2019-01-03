import numpy as np

class Embedder:
    def __init__(self, dim):
        self.dim = dim
        self.word2idx = dict()
        self.weights = None
        self.index = 1

    def create_embeddings(self, list):
        self.weights = np.random.normal(loc=0, scale=0.3, size=(len(list), self.dim))
        self.word2idx = dict(zip(list, range(0, len(list))))

    def load_embeddings(self, file_name, sep=' '):
        weights = []
        weights.append(np.zeros(self.dim))
        with open('./data/' + file_name, 'r') as file:
            for line in file:
                values = line.split(sep)
                if values[0] not in self.word2idx and len(values[1:]) == self.dim:
                    self.word2idx[values[0]] = self.index
                    weights.append(np.array(values[1:], dtype=np.float32))
                    self.index += 1

        if self.weights is None:
            self.weights = np.array(weights)
        else:
            self.weights = np.append(self.weights, weights, axis=0)

    def create_indexed_sentences(self, sentences, pad_size=70):
        indexed = list(map(lambda sent: [self.word2idx.get(word, 0) for word in sent], sentences))
        indexed = [i + [0] * (pad_size - len(i)) for i in indexed]
        return np.array(indexed)

    def create_embedded_sentences(self, sentences, pad_size=70):
        indexed = list(map(lambda sent: [self.word2idx[word] for word in sent], sentences))
        # embeddings = list(map(lambda sent: np.asarray([self.weights[index] for index in sent], dtype=np.float32), indexed))
        embeddings = []
        for sentence in indexed:
            to_pad = pad_size - len(sentence)
            sentence = sentence + ([0] * to_pad)
            embeddings.append(np.asarray([self.weights[index] for index in sentence]))

        return np.array(embeddings)
