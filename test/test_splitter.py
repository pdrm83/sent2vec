import os
import re

from sent2vec.constants import DATA_DIR
from sent2vec.splitter import Splitter

def test_sent2words():
    sentences = [
        "Alice is in the Wonderland.",
        "Alice is not in the Wonderland.",
    ]
    splitter = Splitter()
    splitter.sent2words(sentences=sentences, remove_stop_words=['not'])
    assert splitter.words == [['alice', 'wonderland'], ['alice', 'not', 'wonderland']]


def test_text2sents():
    texts = ["This is an awesome book to learn NLP. DistilBERT is an amazing NLP model. We can interchangeably use "
             "embedding, encoding, or vectorizing."]
    splitter = Splitter()
    splitter.text2sents(texts=texts)
    assert str(splitter.sentences[0]) == 'This is an awesome book to learn NLP.'


def test_text2words_01():
    texts = ["This is an awesome book to learn NLP. DistilBERT is an amazing NLP model. We can interchangeably use " \
             "embedding, encoding, or vectorizing."]
    splitter = Splitter()
    splitter.text2words(texts=texts[0])
    assert splitter.words[0] == 'awesome'


def test_text2words_02():
    file_name = os.path.join(DATA_DIR, 'ensemble_method.txt')
    with open(file_name, 'r') as file:
        texts = file.read().replace('\n', '')

    splitter = Splitter()
    splitter.text2words(texts=texts)
    assert len(splitter.words) == 582
    assert splitter.words[0] == 'write'


def test_text2words_03():
    # TODO
    file_name = os.path.join(DATA_DIR, 'negotiation_tips.txt')
    with open(file_name, 'r') as file:
        texts = file.read().replace('\n', '')

    def cleanerizer(texts):
        text_1 = re.sub(r"[(\[].*?[)\]]", "", texts)
        text_2 = re.sub(r'-', r'', text_1)
        return text_2

    splitter = Splitter()
    splitter.text2words(texts=cleanerizer(texts))
    print(splitter.words)
    splitter.text2words(texts=texts)
    print(splitter.words)
