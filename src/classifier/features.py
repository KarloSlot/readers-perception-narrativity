import csv
from os import path
import pandas as pd

if path.exists('../data/BookNLP/'):
    BOOK_PATH = '../data/BookNLP/'
else:
    BOOK_PATH = '../../data/BookNLP/'

print('\n----\nUsing the BookNLP path:', BOOK_PATH, "\n----\n")


def get_POS_str(fname):
    """
    Returns a string for all part-of-speech tags in the given filename.
    """
    df = pd.read_csv(BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['POS_tag'].tolist())


def get_dep_str(fname):
    """
    Returns a string for all dependency tags in the given filename.
    """
    df = pd.read_csv(BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['dependency_relation'].tolist())


def get_words_str(fname):
    """
    Returns a string for all 'word' tokens in the given BookNLP DataFrame.
    """
    df = pd.read_csv(BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['word'].tolist())
