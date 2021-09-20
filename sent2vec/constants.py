import os

ROOT_DIR = os.getcwd() 
TEST_DIR = os.path.join(ROOT_DIR, 'test')
DATA_DIR = os.path.join(TEST_DIR, 'dataset')

ROOT_LOCAL = '/Users/pedramataee/'
ROOT_LINUX = '~'

WIKI_PATH = os.path.join(ROOT_LINUX, 'gensim-data/glove-wiki-gigaword-300') 
PRETRAINED_VECTORS_PATH_WIKI = os.path.join(WIKI_PATH, 'glove-wiki-gigaword-300.gz')

FASTTEXT_NEWS_PATH = os.path.join(ROOT_LINUX, 'gensim-data/fasttext-wiki-news-subwords-300') 
PRETRAINED_VECTORS_PATH_FASTTEXT = os.path.join(FASTTEXT_NEWS_PATH, 'fasttext-wiki-news-subwords-300.gz')