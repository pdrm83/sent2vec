# A Generic Sentence Embedding Library

In the past, we mostly encode text data using, for example, one-hot, term frequency, or TF-IDF (normalized term 
frequency). There are many challenges to these techniques. In recent years, the latest advancements give us the
opportunity to encode sentences or words in more meaningful formats. The **word2vec** technique and BERT language model
are two important ones.

The sentence embedding is an important step of various NLP tasks such as sentiment analysis and summarization. **A 
flexible sentence embedding library is needed to prototype fast and contextualized.** The open-source sent2vec Python 
package gives you the opportunity to do so. You currently have access to the standard encoders. More advanced 
techniques will be added in the later releases. Hope you can use this library in your exciting NLP projects.

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
If you want to use the `BERT` language model (more specifically, `distilbert-base-uncased`) to encode sentences for 
downstream applications, you must use the code below. 
```python
from sent2vec.vectorizer import Vectorizer

sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]
vectorizer = Vectorizer()
vectorizer.bert(sentences)
vectors = vectorizer.vectors
```
Now, you can compute distance among sentences by using their vectors. In the example, as expected, the distance between
`vectors[0]` and `vectors[1]` is less than the distance between `vectors[0]` and `vectors[2]`.

```python
from scipy import spatial

dist_1 = spatial.distance.cosine(vectors[0], vectors[1])
dist_2 = spatial.distance.cosine(vectors[0], vectors[2])
print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
assert dist_1 < dist_2
# dist_1: 0.043, dist_2: 0.192
```

If you want to use a word2vec approach instead, you must first split sentences into lists of words using the 
`sent2words` method from the `Splitter` class. In this stage, you can customize the list of stop-words by adding or 
removing to/from the default list. When you extract the most important words in sentences, you can compute the sentence
embeddings using the `word2vec` method from the `Vectorizer` class. This method computes the average of vectors 
corresponding to the remaining words using the code below. 

```python
from sent2vec.vectorizer import Vectorizer
from sent2vec.splitter import Splitter

sentences = [
    "Alice is in the Wonderland.",
    "Alice is not in the Wonderland.",
]

splitter = Splitter()
splitter.sent2words(sentences=sentences, remove_stop_words=['not'], add_stop_words=[])
# print(splitter.words)
# [['alice', 'wonderland'], ['alice', 'not', 'wonderland']]
vectorizer = Vectorizer()
vectorizer.word2vec(splitter.words, pretrained_vectors_path= MODEL_PATH)
vectors = vectorizer.vectors
```
As seen above, you can use different word2vec models by sending its path to the `word2vec` method. You can use a 
pre-trained model or a customized one.  

And, that's pretty much it!

