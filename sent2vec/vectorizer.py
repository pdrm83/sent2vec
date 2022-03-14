import numpy as np
import os
import gensim
import torch
import transformers as ppb
from ..sent2vec.splitter import *

class Vectorizer:
    def __init__(self, model = 'bert', 
                       pretrained_weights = 'distilbert-base-uncased', 
                       pretrained_vectors_path = None, 
                       ensemble_method = 'average', 
                       remove_stop_words = ['not'], 
                       add_stop_words = []):
        self.vectors = []
        if model == 'bert':
            print(f'Initializing Bert {pretrained_weights}!')
            self.vectorizer = BertVectorizer(pretrained_weights)
        elif model == 'word2vec':
            assert pretrained_vectors_path, 'Need to pass a valid path to load word2vec'
            print('Initializing word2vec!')
            self.vectorizer = GensimVectorizer(pretrained_vectors_path, 
                                               ensemble_method, 
                                               remove_stop_words, 
                                               add_stop_words)
        else:
            raise  TypeError(f'Wrong model name {model} passed.')

    def run(self, sentences):
        # SANITY CHECK
        assert type(sentences) == list, 'A list must be passed!'
        for sentence in sentences:
            if type(sentence) != str:
                raise TypeError(f'All items must be string type but {sentence} is type {type(sentence)}.')
        # RUN
        vectors = self.vectorizer._execute(sentences)
        for idx in range(vectors.shape[0]):
            self.vectors.append(vectors[idx])


class BaseVectorizer():
    def __init__(self, **kwargs):
        self.pretrained_weights = kwargs.get('pretrained_weights')
        self.pretrained_vectors_path = kwargs.get('pretrained_vectors_path')
        self.ensemble_method = kwargs.get('ensemble_method')
        self.remove_stop_words = kwargs.get('remove_stop_words')
        self.add_stop_words = kwargs.get('add_stop_words')
    
    def _load_model(self):
        pass

    def _execute(self):
        pass


class BertVectorizer(BaseVectorizer):
    def __init__(self, pretrained_weights):
        super().__init__(pretrained_weights=pretrained_weights)
        self._load_model()
    
    def _load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Vectorization done on {self.device} device')
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            self.pretrained_weights)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
    
    def _execute(self, sentences):
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
        # attention_mask = torch.tensor(np.where(padded != 0, 1, 0)).type(torch.LongTensor)
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        # Move vector results back to cpu if calculation was done on GPU
        vectors = last_hidden_states[0][:, 0, :].cpu().numpy()
        return vectors

class GensimVectorizer(BaseVectorizer):
    def __init__(self, pretrained_vectors_path, ensemble_method, remove_stop_words, add_stop_words):
        super().__init__(pretrained_vectors_path=pretrained_vectors_path, ensemble_method=ensemble_method, 
                        remove_stop_words=remove_stop_words, add_stop_words=add_stop_words)
        self.splitter = Splitter()
        self._load_model()
    
    def _load_model(self):
        assert self.pretrained_vectors_path, 'Need to pass a valid path to load word2vec'
        _, file_extension = os.path.splitext(self.pretrained_vectors_path)
        # Checks if file extension is binary
        if file_extension == '.bin':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrained_vectors_path, binary=True)
        elif file_extension == '.txt' or '.gz':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrained_vectors_path)
        else:
            raise IOError(f'The file extension {file_extension} is not valid. Word2vec valid formats are ".txt" and ".bin".')
    
    def _execute(self, sentences):
        self.splitter.sent2words(sentences=sentences, remove_stop_words=self.remove_stop_words, add_stop_words=self.add_stop_words)
        words = self.splitter.words
        vectors = []
        for element in words:
            temp = []
            for w in element:
                temp.append(self.model[w])
            if self.ensemble_method == 'average':
                element_vec = np.mean(temp, axis=0)
                try:
                    vectors.any()
                except:
                    vectors = element_vec
                else:
                    vectors = np.vstack([vectors, element_vec])
        return vectors
        