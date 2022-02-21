import numpy as np
import os
import gensim
import torch
import transformers as ppb


class Vectorizer:
    def __init__(self, model='bert', pretrained_weights='distilbert-base-uncased', pretrained_vectors_path=None, ensemble_method='average'):
        self.vectors = []
        self.use_bert = True
        if model == 'bert':
            print(f'Initializing Bert {pretrained_weights}!')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Vectorization done on {device} device')
            model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                                ppb.DistilBertTokenizer,
                                                                pretrained_weights)
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
        elif model == 'word2vec':
            print('Initializing word2vec!')
            self.use_bert = False
            self.ensemble_method = ensemble_method
            _, file_extension = os.path.splitext(pretrained_vectors_path)
            # Checks if file extension is binary
            if file_extension == '.bin':
                self.model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vectors_path, binary=True)
            elif file_extension == '.txt':
                self.model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vectors_path)
            else:
                raise IOError(f'The file extension {file_extension} is not valid. Word2vec valid formats are ".txt" and ".bin".')
        else:
            raise  NameError(f'Wrong model name {model} passed.')

    def vectorize(self, input_text):
        ## Insert assertations HERE
        if self.use_bert:
            sentences = input_text
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
            self.vectors = vectors
        else:
            words = input_text
            vectors = []
            for element in words:
                temp = []
                for w in element:
                    temp.append(self.model[w])
                if self.ensemble_method == 'average':
                    vectors.extend([np.mean(temp, axis=0)])

            self.vectors = vectors




    def bert(self, sentences, pretrained_weights='distilbert-base-uncased'):
        # Checks if cuda is available and assigns device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Vectorization done on {device} device')
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        # Move model to device
        model = model.to(device)
        model.eval()
        tokenized = list(map(lambda x: tokenizer.encode(x, add_special_tokens=True), sentences))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        # Move inputs to same device as model
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor).to(device)
        # attention_mask = torch.tensor(np.where(padded != 0, 1, 0)).type(torch.LongTensor)

        with torch.no_grad():
            last_hidden_states = model(input_ids)

        # Move vector results back to cpu if calculation was done on GPU
        vectors = last_hidden_states[0][:, 0, :].cpu().numpy()
        self.vectors = vectors

    def word2vec(self, words, pretrained_vectors_path, ensemble_method='average'):
        _, file_extension = os.path.splitext(pretrained_vectors_path)
        # Checks if file extension is binary
        if file_extension == '.bin':
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vectors_path, binary=True)
        elif file_extension == '.txt':
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vectors_path)
        else:
            raise IOError(f'The file extension {file_extension} is not valid. Word2vec valid formats are ".txt" and ".bin".')

        vectors = []
        for element in words:
            temp = []
            for w in element:
                temp.append(model[w])
            if ensemble_method == 'average':
                vectors.extend([np.mean(temp, axis=0)])

        self.vectors = vectors
