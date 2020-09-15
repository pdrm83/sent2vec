# A Generic Sentence Embedding Library

In natural language processing, we need to encode text data. In the past, we mostly use encoders such as one-hot, 
term frequency, or TF-IDF (normalized term frequency). There are many challenges with these techniques. In the recent 
years, the latest advancements give us opportunity to encode sentences or words in more meaningful formats. The word2vec 
technique and BERT language model are two important ones.

The **sentence embedding** is an important step of many NLP projects from sentiment analysis to summarization. We 
believe that a flexible sentence embedding library is needed to build prototypes fast. That is why we have initiated this 
project. In the early releases, you will have access to the standard encoders. We will add more curated techniques in 
the later releases. Hope you can use this library in your exciting NLP projects.

## Install
The `sent2vec` is developed to help you prototype faster. That is why it has many dependencies on other libraries. The 
module requires the following libraries:

* gensim  
* numpy
* spacy  
* transformers  
* torch  

Then, it can be installed using pip:
```python
pip3 install sent2vec
```

## Usage
If you want to use the the `BERT` language model (more specifically, `distilbert-base-uncased`) to encode sentences for 
downstream applications, you must use the code below. Now, you can compute distance among sentences by using their 
representational vectors. In the example, as expected, the distance between `vectors[0]` and `vectors[1]` is less than 
the distance between `vectors[0]` and `vectors[2]`.

```python
from sent2vec.vectorizer import Vectorizer

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
dist_1: 0.043, dist_2: 0.192
```

If you want to use a word2vec approach instead, you must first split sentences to lists of words using the 
`sent2words` method. In this stage, you can customized the list of stop-words by adding or removing to/from the default
list. When you extract the most important words in sentences, you can compute the sentence embeddings using the `w2v`
method. This method computes the average of vectors corresponding to the remaining words using the code bleow. 
```python
from sent2vec.vectorizer import Vectorizer

sentences = [
    "Alice is in the Wonderland.",
    "Alice is not in the Wonderland.",
]
vectorizer = Vectorizer()
words = vectorizer.sent2words(sentences, remove_stop_words=['not'], add_stop_words=[])
model_path = os.path.join(os.path.abspath(os.getcwd()), 'glove-wiki-gigaword-300')
vectors = vectorizer.w2v(words, model_path= model_path)

print(words)
[['alice', 'wonderland'], ['alice', 'not', 'wonderland']]
```
As seen above, you can use different word2ved model by sending its path to the `w2v` method. You can use a pre-trained
model or a customized one.  

And, that's pretty much it!

