import numpy as np

import gensim
import torch
import transformers as ppb


class Vectorizer:
    def __init__(self):
        self.vectors = []

    def bert(self, sentences, pretrained_weights='distilbert-base-uncased'):
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = list(map(lambda x: tokenizer.encode(x, add_special_tokens=True), sentences))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)
        # attention_mask = torch.tensor(np.where(padded != 0, 1, 0)).type(torch.LongTensor)

        with torch.no_grad():
            last_hidden_states = model(input_ids)

        vectors = last_hidden_states[0][:, 0, :].numpy()
        self.vectors = vectors

    def word2vec(self, words, pretrained_vectors_path, ensemble_method='average'):
        model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vectors_path)

        vectors = []
        for element in words:
            temp = []
            for w in element:
                temp.append(model[w])
            if ensemble_method == 'average':
                vectors.extend([np.mean(temp, axis=0)])

        self.vectors = vectors
