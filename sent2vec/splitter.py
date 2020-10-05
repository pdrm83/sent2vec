import spacy

nlp = spacy.load("en_core_web_sm")
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


class Splitter:
    def __init__(self):
        self.words = []

    def sent2words(self, sentences, **kwargs):
        add_stop_words = kwargs.get('add_stop_words', [])
        remove_stop_words = kwargs.get('remove_stop_words', [])

        for w in add_stop_words:
            nlp.vocab[w].is_stop = True
        for w in remove_stop_words:
            nlp.vocab[w].is_stop = False

        words = []
        for sentence in sentences:
            doc = nlp(sentence.lower())
            words.append([token.lemma_ for token in doc if not token.is_punct | token.is_space | token.is_stop])

        self.words = words
