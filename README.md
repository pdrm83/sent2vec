[![license](https://img.shields.io/badge/license-MIT-success)](https://github.com/pdrm83/Sent2Vec/blob/master/LICENSE.md)
[![doc](https://img.shields.io/badge/docs-Medium-blue)](https://towardsdatascience.com/how-to-compute-sentence-similarity-using-bert-and-word2vec-ab0663a5d64)

# Sent2Vec 
## How to Compute Sentence Embedding Fast and Flexible

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

## Documentation

*class* **sent2vec.vectorizer.Vectorizer**(pretrained_weights='distilbert-base-uncased', ensemble_method='average')

### **Parameters**

- **pretrained_weights**: str, *default*='distilbert-base-uncased' - If the string does not include an extension .txt, .gz or .bin, then Bert vectorizer is loaded using the specified weights. *Example: pass 'distilbert-base-multilingual-cased' to load Bert base multilingual model.* <br/> To load word2vec vectorizer pass a valid path to the weights file (.txt, .gz or .bin). *Example: pass 'glove-wiki-gigaword-300.gz' to load the Wiki vectors (when saved in the same folder you are running the code).*
- **ensemble_method**: str, *default*='average' - How word vectors are computed into sentece vectors.

### **Methods**

run(sentences, remove_stop_words = ['not'], add_stop_words = [])
- **sentences**: list, - List of sentences.
- **remove_stop_words**: list, *default*=['not'] - When using sent2vec, list of words to remove from *stop words* when splitting sentences.
- **add_stop_words**: list, *default*=[] - When using sent2vec, list of words to add to *stop words* when splitting sentences.

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
vectorizer.run(sentences)
vectors = vectorizer.vectors
```
Default Vectorizer weights are `distilbert-base-uncased` but it's possible to pass the argument `pretrained_weights` to chose another `BERT` model.

For example, to load `BERT base multilingual model`:

```python
vectorizer = Vectorizer(pretrained_weights='distilbert-base-multilingual-cased')
```

Now it's possible to compute distance among sentences by using their vectors. In the example, as expected, the distance between
`vectors[0]` and `vectors[1]` is less than the distance between `vectors[0]` and `vectors[2]`.

```python
from scipy import spatial

dist_1 = spatial.distance.cosine(vectors[0], vectors[1])
dist_2 = spatial.distance.cosine(vectors[0], vectors[2])
print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
assert dist_1 < dist_2
# dist_1: 0.043, dist_2: 0.192
```

If you want to use a word2vec approach instead, you must pass a valid path to the model weights. Under the hood the sentences will be splitted into lists of words using the `sent2words` method from the `Splitter` class. It is possible to customize the list of stop-words by adding or removing to/from the default list. Two additional arguments (both lists) must be passed when the vectorizer's method .run is called: `remove_stop_words` and `add_stop_words`. 

NOTE: When you extract the most important words in sentences, by default `Vectorizer` computes the sentence embeddings using the average of vectors corresponding to the remaining words. 

```python
from sent2vec.vectorizer import Vectorizer

sentences = [
    "Alice is in the Wonderland.",
    "Alice is not in the Wonderland.",
]

vectorizer = Vectorizer(pretrained_weights= PRETRAINED_VECTORS_PATH)
vectorizer.run(sentences, remove_stop_words=['not'], add_stop_words=[])
vectors = vectorizer.vectors
```

And, that's pretty much it!