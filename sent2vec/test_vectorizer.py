from sent2vec.vectorizer import Vectorizer
from scipy.spatial import distance


def test_sent2vec_bert_01():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP library.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer(sentences)
    vectors = vectorizer.sent2vec_bert()
    assert len(vectors[0, :]) == 768


def test_sent2vec_bert_02():
    sentences = [
        "This is an awesome book to learn NLP.",
        "DistilBERT is an amazing NLP library.",
        "We can interchangeably use embedding, encoding, or vectorizing.",
    ]
    vectorizer = Vectorizer(sentences)
    vectors = vectorizer.sent2vec_bert()
    dist_1 = distance.cosine(vectors[0], vectors[1])
    dist_2 = distance.cosine(vectors[0], vectors[2])
    assert dist_1 < dist_2

