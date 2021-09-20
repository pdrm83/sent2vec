import os
import re

import spacy

os.environ['LANGUAGE_MODEL_SPACY'] = "en_core_web_md"
nlp = spacy.load(os.environ['LANGUAGE_MODEL_SPACY'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


class Splitter:
    def __init__(self):
        self.words = []
        self.sentences = []

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

    def text2sents(self, texts):
        for text in texts:
            doc = nlp(text)
            span = doc[0:5]
            span.merge()
            sents = list(doc.sents)
            self.sentences.extend([sent for sent in sents])

    def text2words(self, texts):
        doc = nlp(texts)
        tokenized_texts = []
        for w in doc:
            is_clean = w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num
            if is_clean:
                tokenized_texts.append(w.lemma_)

        self.words = tokenized_texts


def sentencizer_by_regex(texts):
    alphabets = "([A-Za-z])"
    prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = r"(Inc|Ltd|Jr|Sr|Co|etc)"
    starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = r"[.](com|net|org|io|gov)"

    text = " " + texts + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
