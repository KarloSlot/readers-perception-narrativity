import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import features

TMV_DF_file = 'Universal_Annotation_Results_Selection.csv'
try:
    TMV_DF = pd.read_csv('../data/' + TMV_DF_file)
except FileNotFoundError:
    TMV_DF = pd.read_csv('../../data/' + TMV_DF_file)
MAX_FEATURES = None


def pos_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using part-of-speech unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(1,1), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(1,1), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)

    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names()

    else:
        return X_train, X_test


def pos_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(2,2), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(2,2), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(3,3), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(3,3), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))

    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech bigrams and trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(2,3), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(2,3), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names()
    else:
        return X_train, X_test


def dep_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def word_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names(), vectorizer
    else:
        return X_train, X_test


def word_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def word_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def word_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW bigrams and trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test
