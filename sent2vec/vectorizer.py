import numpy as np
import os

import gensim
import spacy
import torch
import transformers as ppb


class Vectorizer:
    def __init__(self):
        self.sentences = []
        self.words = []
        self.vectors = []

    def bert(self, sentences, pretrained_weights='distilbert-base-uncased'):
        self.sentences = sentences
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        encoder = lambda x: tokenizer.encode(x, add_special_tokens=True)
        tokenized = list(map(encoder, sentences))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)
        attention_mask = torch.tensor(attention_mask).type(torch.LongTensor)

        with torch.no_grad():
            last_hidden_states = model(input_ids)

        vectors = last_hidden_states[0][:, 0, :].numpy()

        self.vectors = vectors
        return vectors

    def w2v(self, words, **kwargs):
        self.words = words
        model_path_default = os.path.join(os.path.abspath(os.getcwd()), 'glove-wiki-gigaword-300')

        model_path = kwargs.get('model_path', model_path_default)
        ensemble_method = kwargs.get('ensemble_method', 'mean')

        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)

        vectors = []
        for element in words:
            temp = []
            for w in element:
                temp.append(model[w])
            if ensemble_method == 'mean':
                vectors.extend([np.mean(temp, axis=0)])

        return vectors

    @staticmethod
    def sent2words(sentences, **kwargs):
        add_stop_words = kwargs.get('add_stop_words', [])
        remove_stop_words = kwargs.get('remove_stop_words', [])
        language_model = kwargs.get('language_model', 'en_core_web_sm')

        nlp = spacy.load(language_model)
        for w in add_stop_words:
            nlp.vocab[w].is_stop = True
        for w in remove_stop_words:
            nlp.vocab[w].is_stop = False

        words = []
        for sentence in sentences:
            doc = nlp(sentence.lower())
            words.append([token.lemma_ for token in doc if not token.is_punct | token.is_space | token.is_stop])

        return words
