import numpy as np
import pandas as pd

import torch
import transformers as ppb


class Vectorizer:
    def __init__(self, sentences):
        """
            sentences: A list of strings or sentences
        """
        self.batch = pd.DataFrame(sentences)

    def sent2vec_bert(self):
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            'distilbert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = self.batch[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(np.array(padded))
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids)

        features = last_hidden_states[0][:, 0, :].numpy()

        return features

