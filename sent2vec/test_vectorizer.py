from sent2vec.vectorizer import Vectorizer

from numpy import dot
from numpy.linalg import norm


def cosine_distance(a, b):
    return 1 - dot(a, b) / (norm(a) * norm(b))


def test_sent2vec_bert_01():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding or encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    vectors = vectorizer.bert(sentences)
    assert len(vectors[0, :]) == 768


def test_sent2vec_bert_02():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    vectors = vectorizer.bert(sentences)
    dist_1 = cosine_distance(vectors[0], vectors[1])
    dist_2 = cosine_distance(vectors[0], vectors[2])
    print('dist_1: {}'.format(dist_1), 'dist_2: {}'.format(dist_2))
    assert dist_1 < dist_2


def test_sent2vec_w2v():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP model.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer()
    words = vectorizer.sent2words(sentences, add_stop_words=['distilbert', 'vectorizing'])
    vectors = vectorizer.w2v(words)
    dist_1 = cosine_distance(vectors[0], vectors[1])
    dist_2 = cosine_distance(vectors[0], vectors[2])
    assert dist_1 < dist_2


def test_sent2vec_w2v_bert():
    sentences = [
        "Alice is in the Wonderland.",
        "Alice is not in the Wonderland.",
    ]
    vectorizer = Vectorizer()

    vectors_bert = vectorizer.bert(sentences)
    dist_bert = cosine_distance(vectors_bert[0], vectors_bert[1])

    words = vectorizer.sent2words(sentences, remove_stop_words=['not'])
    vectors_w2v = vectorizer.w2v(words)
    dist_w2v = cosine_distance(vectors_w2v[0], vectors_w2v[1])

    assert dist_w2v > dist_bert


def test_sent2words():
    sentences = [
        "Alice is in the Wonderland.",
        "Alice is not in the Wonderland.",
    ]
    vectorizer = Vectorizer()
    words = vectorizer.sent2words(sentences=sentences, remove_stop_words=['not'])
    assert words == [['alice', 'wonderland'], ['alice', 'not', 'wonderland']]

test_sent2vec_w2v()