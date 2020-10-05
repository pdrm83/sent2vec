import os
from scipy import spatial

from sent2vec.vectorizer import Vectorizer
from sent2vec.splitter import Splitter

MODEL_PATH = '/Users/pedramataee/gensim-data/glove-wiki-gigaword-300'
pretrained_vectors_path = os.path.join(MODEL_PATH, 'glove-wiki-gigaword-300')


def test_vectorizer_bert_01():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding or encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    assert len(vectorizer.vectors[0, :]) == 768


def test_vectorizer_bert_02():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    dist_1 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    dist_2 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[2])
    print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
    assert dist_1 < dist_2


def test_vectorizer_word2vec():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    splitter = Splitter()
    splitter.sent2words(sentences, add_stop_words=['distilbert', 'vectorizing'])
    vectorizer = Vectorizer()
    vectorizer.word2vec(splitter.words, pretrained_vectors_path=pretrained_vectors_path)
    dist_1 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    dist_2 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[2])
    assert dist_1 < dist_2


def test_vectorizer_quality():
    sentences = [
        "Alice is in the Wonderland.",
        "Alice is not in the Wonderland.",
    ]
    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    vectors_bert = vectorizer.vectors
    dist_bert = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])

    splitter = Splitter()
    splitter.sent2words(sentences=sentences, remove_stop_words=['not'], add_stop_words=[])
    vectorizer.word2vec(splitter.words, pretrained_vectors_path=pretrained_vectors_path)
    vectors_w2v = vectorizer.vectors
    dist_w2v = spatial.distance.cosine(vectors_w2v[0], vectors_w2v[1])

    print('dist_bert: {0}, dist_w2v: {1}'.format(dist_bert, dist_w2v))
    assert dist_w2v > dist_bert


def test_splitter_sent2words():
    sentences = [
        "Alice is in the Wonderland.",
        "Alice is not in the Wonderland.",
    ]
    splitter = Splitter()
    splitter.sent2words(sentences=sentences, remove_stop_words=['not'])
    assert splitter.words == [['alice', 'wonderland'], ['alice', 'not', 'wonderland']]


test_vectorizer_quality()