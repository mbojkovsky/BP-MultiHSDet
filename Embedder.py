import numpy as np

class Embedder:
    def __init__(self, dim, file):
        self.dim = dim
        self.file = file
        self.word2idx = dict()
        self.weights = []

    def load_embeddings(self):
        index = 1
        self.weights.append(np.zeros(1024))
        with open('./data/' + self.file, 'r') as file:
            for line in file:
                values = line.split('\t')
                if values[0] not in self.word2idx and len(values[1:]) == 1024:
                    self.word2idx[values[0]] = index
                    self.weights.append(values[1:])
                    index += 1

        self.weights = np.asarray(self.weights)

    def create_indexed_sentences(self, sentences, pad_size=70):
        indexed = list(map(lambda sent: [self.word2idx[word] for word in sent], sentences))
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
