import os
from scipy import spatial
import sys

from sent2vec.constants import PRETRAINED_VECTORS_PATH_WIKI, ROOT_DIR
from sent2vec.vectorizer import Vectorizer, BertVectorizer
from sent2vec.splitter import Splitter


def test_bert_01():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding or encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    vectorizer.run(sentences)
    assert len(vectorizer.vectors[0]) == 768


def test_bert_02():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    vectorizer.run(sentences)
    dist_1 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    dist_2 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[2])
    print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
    assert dist_1 < dist_2
    

def test_bert_03():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer(pretrained_weights="bert-base-multilingual-cased")
    vectorizer.run(sentences)
    dist_1 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    dist_2 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[2])
    print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
    assert dist_1 < dist_2


def test_bert_04():
    sentences = [
        "401k retirement accounts",
        "401k retirement accounts"
    ]
    vectorizer = Vectorizer()
    vectorizer.run(sentences)
    dist = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    assert dist == 0


def test_bert_05():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
]
    new_sentences = [
        "这是一本学习 NLP 的好书",
        "DistilBERT 是一个了不起的 NLP 模型",
        "我们可以交替使用嵌入、编码或矢量化。",
    ]
    vectorizer = Vectorizer(pretrained_weights="bert-base-multilingual-cased")
    vectorizer.run(sentences)
    vectorizer.run(new_sentences)
    vectors = vectorizer.vectors
    assert len(vectors) == 6


def test_word2vec():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    
    vectorizer = Vectorizer(pretrained_weights= PRETRAINED_VECTORS_PATH_WIKI)
    vectorizer.run(sentences, add_stop_words=['distilbert', 'vectorizing'])

    dist_1 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[1])
    dist_2 = spatial.distance.cosine(vectorizer.vectors[0], vectorizer.vectors[2])
    assert dist_1 < dist_2


def test_complete():
    sentences = [
        "Alice is in the Wonderland.",
        "Alice is not in the Wonderland.",
    ]
    vectorizer = Vectorizer()
    vectorizer.run(sentences)
    vectors_bert = vectorizer.vectors
    dist_bert = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])

    vectorizer = Vectorizer(pretrained_weights= PRETRAINED_VECTORS_PATH_WIKI)
    vectorizer.run(sentences)
    vectors_w2v = vectorizer.vectors
    dist_w2v = spatial.distance.cosine(vectors_w2v[0], vectors_w2v[1])

    print('dist_bert: {0}, dist_w2v: {1}'.format(dist_bert, dist_w2v))
    assert dist_w2v > dist_bert
