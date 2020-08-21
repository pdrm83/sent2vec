# A Generic Sentence Embedding Library

In natural language processing, we need to encode text data. In the past, we mostly use encoders such as one-hot, 
term frequency, or TF-IDF (normalized term frequency). There are many challenges with these techniques. In the recent 
years, the latest advancements give us opportunity to encode sentences or words in more meaningful formats. The word2vec 
technique and BERT language model are two important ones.

The **sentence embedding** is an important step of many NLP projects from sentiment analysis to summarization. We 
believe that a flexible sentence embedding library is needed to build prototypes fast. That is why we have initiated this 
project. In the early releases, you will have access to the standard encoders. We will add more curated techniques in 
the later releases. Hope you can use this library in your exciting NLP projects.

## Library
The package requires the following libraries:

* gensim  
* numpy
* spacy  
* transformers  
* torch  

The `sent2vec` package is developed to help you prototype faster. That is why it has many dependencies on other 
libraries.

## Install

It can be installed using pip:
```python
pip3 install sent2vec
```

## Usage
If you want to use the the `BERT` language model (more specifically, `distilbert-base-uncased`) to compute sentence 
embedding, you must use the code below. 

```python
from sent2vec.vectorizer import Vectorizer

sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]
vectorizer = Vectorizer()
vectors = vectorizer.bert(sentences)
```
Having the corresponding vectors, you can compute distance among vectors. Here, as expected, the distance between 
`vectors[0]` and `vectors[1]` is less than the distance between `vectors[0]` and `vectors[2]`.

```python
dist_1 = cosine_distance(vectors[0], vectors[1])
dist_2 = cosine_distance(vectors[0], vectors[2])

print('dist_1: {}'.format(dist_1), 'dist_2: {}'.format(dist_2))
dist_1: 0.043, dist_2: 0.192
```

If you want to use a `word2vec` approach instead, you must first split sentences to lists of words using the 
`sent2words` method. In this stage, you can customized the list of stop-words by adding or removing to/from the default
list. When you extract the most important words in sentences, you can compute the sentence embeddings using the `w2v`
method. This method computes the average of vectors corresponding to the remaining words using the code bleow. 
```python
sentences = [
    "Alice is in the Wonderland.",
    "Alice is not in the Wonderland.",
]
model_path = os.path.join(os.path.abspath(os.getcwd()), 'glove-wiki-gigaword-300')
vectorizer = Vectorizer()
words = vectorizer.sent2words(sentences, remove_stop_words=['not'], add_stop_words=[])
vectors = vectorizer.w2v(words, model_path= model_path)
```
As you can see above, you can use different `word2ved` model by sending it to the `w2v` method. 

And, that's pretty much it!

