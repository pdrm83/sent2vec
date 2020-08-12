# A Generic Sentence Embedding Library

In natural language processing, we need to vectorize or encode text data to let machine proocess it. In the past, we 
mostly use encoders such as one-hot, term frequency, and TF-IDF (normalized term frequency). There are many challenges 
with these techniques that you can read here. In the recent years, the deep learning advancements give us opportunity to 
encode sentences or words in more meaningful format. The word2vec library was one of the major advancement in this
field. Or, the BERT language model provides us a powerful sentence encoders that can be used in many projects. 

The sentence embedding or encoding is an important step of many NLP projects. Plus, we believe that a flexible sent2vec
library is needed to build a prototype fast. That is why we have initiated this project. In the early releases, you will
have access to the standard encoders. We will add more curated techniques in the later releases. Hope you can use this 
library in your exciting NLP projects.  

## Library
The library requires the following libraries:

* transformers
* pandas
* numpy
* torch

## Install

It can be installed using pip:
```python
pip install sent2vec
```

## Usage

This is how to initialize the library and provide the data.
```python
from sent2vec.vectorizer import Vectorizer

sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP library.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]
vectorizer = Vectorizer(sentences)
```

If you want to use the pre-trained DistilBertModel, you should use the code below. 
```python
vectors = vectorizer.sent2vec_bert()
```

And, that's pretty much it!

