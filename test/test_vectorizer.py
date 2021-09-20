import os
from scipy import spatial

from sent2vec.constants import PRETRAINED_VECTORS_PATH_WIKI
from sent2vec.vectorizer import Vectorizer
from sent2vec.splitter import Splitter

def test_bert_01():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding or encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    assert len(vectorizer.vectors[0, :]) == 768


def test_bert_02():
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


def test_bert_03():
    sentences = [
        "401k retirement accounts",
        "401k retirement accounts"
    ]
    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    dist = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    assert dist == 0


def test_bert_04():
    sentences = ["401k retirement accounts"]
    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    vec_1 = vectorizer.vectors[0]
    vectorizer.bert(sentences)
    vec_2 = vectorizer.vectors[0]
    dist = spatial.distance.cosine(vec_1, vec_2)
    assert dist == 0


def test_word2vec():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    splitter = Splitter()
    splitter.sent2words(sentences, add_stop_words=['distilbert', 'vectorizing'])
    vectorizer = Vectorizer()
    vectorizer.word2vec(splitter.words, pretrained_vectors_path=PRETRAINED_VECTORS_PATH_WIKI)
    dist_1 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    dist_2 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[2])
    assert dist_1 < dist_2


def test_complete():
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
    vectorizer.word2vec(splitter.words, pretrained_vectors_path=PRETRAINED_VECTORS_PATH_WIKI)
    vectors_w2v = vectorizer.vectors
    dist_w2v = spatial.distance.cosine(vectors_w2v[0], vectors_w2v[1])

    print('dist_bert: {0}, dist_w2v: {1}'.format(dist_bert, dist_w2v))
    assert dist_w2v > dist_bert


def test_models():
    sentences = ["'Artificial Intelligence: Unorthodox Lessons' is an amazing book to gain insights about AI."]
    splitter = Splitter()
    splitter.sent2words(sentences=sentences)
    vectorizer = Vectorizer()
    vectorizer.word2vec(splitter.words, pretrained_vectors_path=PRETRAINED_VECTORS_PATH_WIKI)
    vectors_wiki = vectorizer.vectors

    sentences = ["'Artificial Intelligence: Unorthodox Lessons' is an amazing book to gain insights about AI."]
    splitter = Splitter()
    splitter.sent2words(sentences=sentences)
    vectorizer = Vectorizer()
    vectorizer.word2vec(splitter.words, pretrained_vectors_path=PRETRAINED_VECTORS_PATH_WIKI)
    vectors_fasttext = vectorizer.vectors

    dist = spatial.distance.cosine(vectors_wiki, vectors_fasttext)
    print(dist)
