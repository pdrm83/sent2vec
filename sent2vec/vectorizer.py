from sent2vec.modules import *


class Vectorizer:
    def __init__(self, model='bert', pretrained_weights='distilbert-base-uncased', pretrained_vectors_path=None, ensemble_method='average'):
        self.vectors = []
        if model == 'bert':
            print(f'Initializing Bert {pretrained_weights}!')
            self.vectorizer = BertVectorizer(pretrained_weights)
        elif model == 'word2vec':
            assert pretrained_vectors_path, 'Need to pass a valid path to load word2vec'
            print('Initializing word2vec!')
            self.vectorizer = GensimVectorizer(pretrained_vectors_path, ensemble_method)
        else:
            raise  TypeError(f'Wrong model name {model} passed.')

    def run(self, sentences):
        vectors = self.vectorizer.execute(sentences)
        for idx in range(vectors.shape[0]):
            self.vectors.append(vectors[idx])
