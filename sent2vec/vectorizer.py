import numpy as np
import os
import gensim
import torch
import transformers as ppb
from sent2vec.splitter import *

class Vectorizer:
    def __init__(self, pretrained_weights = 'distilbert-base-uncased', 
                       ensemble_method = 'average'):
        _, ext = os.path.splitext(pretrained_weights)
        self.vectors = []
        if not ext:
            print(f'Initializing Bert {pretrained_weights}')
            self.vectorizer = BertVectorizer(pretrained_weights=pretrained_weights)
        else:
            print(f'Initializing word2vec with vector path {pretrained_weights}')
            self.vectorizer = GensimVectorizer(pretrained_weights=pretrained_weights, 
                                               ensemble_method=ensemble_method)

    def run(self, sentences, remove_stop_words = ['not'], add_stop_words = []):
        # SANITY CHECK
        assert type(sentences) == list, 'A list must be passed!'
        for sentence in sentences:
            if type(sentence) != str:
                raise TypeError(f'All items must be string type but {sentence} is type {type(sentence)}.')
        # RUN
        vectors = self.vectorizer._execute(sentences, remove_stop_words=remove_stop_words, add_stop_words=add_stop_words)
        for idx in range(vectors.shape[0]):
            self.vectors.append(vectors[idx])


class BaseVectorizer():
    def __init__(self, **kwargs):
        self.pretrained_weights = kwargs.get('pretrained_weights')
        self.ensemble_method = kwargs.get('ensemble_method')
    
    def _load_model(self):
        pass

    def _execute(self):
        pass


class BertVectorizer(BaseVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Vectorization done on {self.device}')
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            self.pretrained_weights)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
    
    def _execute(self, sentences, **kwargs):
        model = self.model.to(self.device)
        model.eval()
        tokenized = list(map(lambda x: self.tokenizer.encode(x, add_special_tokens=True), sentences))
        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        # Move inputs to same device as model
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor).to(self.device)
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        # Move vector results back to cpu if calculation was done on GPU
        vectors = last_hidden_states[0][:, 0, :].cpu().numpy()
        return vectors

class GensimVectorizer(BaseVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.splitter = Splitter()
        self._load_model()
    
    def _load_model(self):
        _, file_extension = os.path.splitext(self.pretrained_weights)
        # Checks if file extension is binary
        if file_extension == '.bin':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrained_weights, binary=True)
        elif file_extension == '.txt' or '.gz':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrained_weights)
        else:
            raise IOError(f'The file extension {file_extension} is not valid. Word2vec valid formats are ".txt" and ".bin".')
    
    def _execute(self, sentences, **kwargs):
        self.splitter.sent2words(sentences, remove_stop_words=kwargs.get('remove_stop_words'), add_stop_words=kwargs.get('add_stop_words'))
        words = self.splitter.words
        vectors = []
        for element in words:
            temp = []
            for w in element:
                temp.append(self.model[w])
            if self.ensemble_method == 'average':
                element_vec = np.mean(temp, axis=0)
                try:
                    vectors = np.vstack([vectors, element_vec])
                except:
                    vectors = element_vec
                    
        return vectors
        